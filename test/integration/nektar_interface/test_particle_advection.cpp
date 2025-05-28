#include "../../unit/nektar_interface/test_helper_utilities.hpp"

// Test advecting particles between ranks
TEST(ParticleGeometryInterface, Advection2D) {

  const int N_total = 1000;
  const double tol = 1.0e-10;

  TestUtilities::TestResourceSession resource_session(
      "square_triangles_quads.xml");

  // Create session reader.
  auto session = resource_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto config = std::make_shared<ParameterStore>();
  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh, config);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int Nsteps = 2000;
  const REAL dt = 0.10;
  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);
    auto velocities =
        NESO::Particles::normal_distribution(N, 3, 0.0, 0.5, rng_vel);
    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
        initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = pos_orig;
      }
      for (int dimx = 0; dimx < 3; dimx++) {
        initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
      const auto px_rank = uniform_dist(rng_rank);
      initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  auto lambda_advect = [&] {
    auto t0 = profile_timestamp();
    particle_loop(
        A,
        [=](auto P, auto V) {
          for (int dimx = 0; dimx < ndim; dimx++) {
            P.at(dimx) += dt * V.at(dimx);
          }
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
        ->execute();
    sycl_target->profile_map.inc("Advect", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };

  auto lambda_check_owning_cell = [&] {
    Array<OneD, NekDouble> global_coord(3);
    Array<OneD, NekDouble> local_coord(3);
    Array<OneD, NekDouble> eta(3);
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A->position_dat->cell_dat.get_cell(cellx);
      auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
      auto reference_positions =
          A->get_cell(Sym<REAL>("NESO_REFERENCE_POSITIONS"), cellx);

      for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {

        const int cell_neso = (*cell_ids)[0][rowx];
        ASSERT_EQ(cell_neso, cellx);
        const int cell_nektar = cell_id_translation.map_to_nektar[cell_neso];

        auto geom = graph->GetGeometry2D(cell_nektar);
        local_coord[0] = reference_positions->at(rowx, 0);
        local_coord[1] = reference_positions->at(rowx, 1);
        global_coord[0] = geom->GetCoord(0, local_coord);
        global_coord[1] = geom->GetCoord(1, local_coord);

        geom->GetXmap()->LocCoordToLocCollapsed(local_coord, eta);
        // check the global coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double err_abs =
              ABS(positions->at(rowx, dimx) - global_coord[dimx]);
          ASSERT_TRUE(err_abs <= tol);
          ASSERT_TRUE(std::fabs((double)eta[dimx]) < (1.0 + tol));
        }
      }
    }
  };

  REAL T = 0.0;

  for (int stepx = 0; stepx < Nsteps; stepx++) {

    pbc.execute();
    A->hybrid_move();
    cell_id_translation.execute();
    A->cell_move();
    lambda_check_owning_cell();

    lambda_advect();

    T += dt;
    // if ((stepx % 100 == 0) && (rank == 0)) {
    //   std::cout << stepx << std::endl;
    // }
  }

  mesh->free();
}

class ParticleAdvection3D
    : public testing::TestWithParam<
          std::tuple<std::string, std::string, double, INT>> {};
TEST_P(ParticleAdvection3D, Advection3D) {
  // Test advecting particles between ranks

  std::tuple<std::string, std::string, double, INT> param = GetParam();

  const int N_total = 2000;
  const double tol = std::get<2>(param);

  TestUtilities::TestResourceSession resource_session(
      static_cast<std::string>(std::get<1>(param)),
      static_cast<std::string>(std::get<0>(param)));

  // Create session reader.
  auto session = resource_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);

  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms_3d;
  get_all_elements_3d(graph, geoms_3d);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  extend_halos_fixed_offset(1, mesh);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto config = std::make_shared<ParameterStore>();

  config->set<REAL>("MapParticlesNewton/newton_tol", 1.0e-10);
  // There are some pyramid corners that are hard to bin into with tighter
  // tolerances.
  config->set<REAL>("MapParticlesNewton/contained_tol", 1.0e-2);
  // Use the non-linear mapper for all geoms
  config->set<INT>("MapParticles3D/all_generic_newton", std::get<3>(param));

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh, config);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 3;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int Nsteps = 1000;
  const REAL dt = 0.1;
  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);
    auto velocities =
        NESO::Particles::normal_distribution(N, 3, 0.0, 0.5, rng_vel);
    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
        initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = pos_orig;
      }
      for (int dimx = 0; dimx < 3; dimx++) {
        initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
      const auto px_rank = uniform_dist(rng_rank);
      initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  auto lambda_advect = [&] {
    auto t0 = profile_timestamp();
    particle_loop(
        A,
        [=](auto P, auto V) {
          for (int dimx = 0; dimx < ndim; dimx++) {
            P.at(dimx) += dt * V.at(dimx);
          }
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
        ->execute();
    sycl_target->profile_map.inc("Advect", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };

  auto lambda_check_owning_cell = [&] {
    Array<OneD, NekDouble> global_coord(3);
    Array<OneD, NekDouble> xi(3);
    Array<OneD, NekDouble> eta(3);
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A->position_dat->cell_dat.get_cell(cellx);
      auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
      auto reference_positions =
          A->get_cell(Sym<REAL>("NESO_REFERENCE_POSITIONS"), cellx);

      const int cell_nektar = cell_id_translation.map_to_nektar[cellx];
      auto geom = geoms_3d[cell_nektar];

      int shape_type = geom->GetShapeType();
      Newton::XMapNewton<Newton::MappingGeneric3D> mapper(sycl_target, geom);

      for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {
        const int cell_neso = (*cell_ids)[0][rowx];
        ASSERT_EQ(cell_neso, cellx);

        xi[0] = reference_positions->at(rowx, 0);
        xi[1] = reference_positions->at(rowx, 1);
        xi[2] = reference_positions->at(rowx, 2);
        global_coord[0] = geom->GetCoord(0, xi);
        global_coord[1] = geom->GetCoord(1, xi);
        global_coord[2] = geom->GetCoord(2, xi);

        // check the global coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double err_abs =
              ABS(positions->at(rowx, dimx) - global_coord[dimx]);
          const double err_rel =
              err_abs > 0.0 ? err_abs / std::abs(global_coord[dimx]) : err_abs;
          ASSERT_TRUE((err_abs <= tol) || (err_rel <= tol));
          ASSERT_TRUE(std::fabs((double)eta[dimx]) < (1.0 + tol));
        }
      }
    }
  };

  REAL T = 0.0;
  for (int stepx = 0; stepx < Nsteps; stepx++) {
    pbc.execute();
    A->hybrid_move();
    cell_id_translation.execute();
    A->cell_move();
    lambda_check_owning_cell();

    lambda_advect();
    T += dt;
  }

  mesh->free();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleMeshes, ParticleAdvection3D,
    testing::Values(std::tuple<std::string, std::string, double, INT>(
                        "reference_all_types_cube/conditions.xml",
                        "reference_all_types_cube/linear_non_regular_0.5.xml",
                        1.0e-4, // The non-linear exit tolerance in Nektar is
                                // like (err_x * err_x
                                // + err_y * err_y) < 1.0e-8
                        0),
                    std::tuple<std::string, std::string, double, INT>(
                        "reference_all_types_cube/conditions.xml",
                        "reference_all_types_cube/mixed_ref_cube_0.5.xml",
                        1.0e-10, 0),
                    std::tuple<std::string, std::string, double, INT>(
                        "reference_all_types_cube/conditions.xml",
                        "reference_all_types_cube/linear_non_regular_0.5.xml",
                        1.0e-4, // The non-linear exit tolerance in Nektar is
                                // like (err_x * err_x
                                // + err_y * err_y) < 1.0e-8
                        1 // Use the Generic3D mapper for the linear geoms.
                        )));

TEST(ParticleAdvection3D, Torus) {
  // Test advecting particles between ranks

  const int N_per_element = 10;
  const double tol = 1.0e-4;

  TestUtilities::TestResourceSession resource_session(
      "torus/quarter_torus_tets_order_2.xml", "torus/conditions.xml");
  const REAL torus_radius_major = 5.0;
  const REAL torus_radius_minor = 1.9;

  // Create session reader.
  auto session = resource_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);

  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms_3d;
  get_all_elements_3d(graph, geoms_3d);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  extend_halos_fixed_offset(1, mesh);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto config = std::make_shared<ParameterStore>();

  config->set<REAL>("MapParticlesNewton/newton_tol", 1.0e-8);
  // There are some pyramid corners that are hard to bin into with tighter
  // tolerances.
  config->set<REAL>("MapParticlesNewton/contained_tol", 1.0e-2);
  // Use the non-linear mapper for all geoms
  config->set<INT>("MapParticles3D/all_generic_newton", 1);

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh, config);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 3;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 2)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);

  const int Nsteps = 400;
  const REAL dt = 0.1;
  const int cell_count = domain->mesh->get_cell_count();

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;

  rng_pos = uniform_within_elements(graph, N_per_element, positions, cells,
                                    1.0e-4, rng_pos);

  const int N = cells.size();

  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 0.5, rng_vel);
  std::uniform_int_distribution<int> uniform_dist(
      0, sycl_target->comm_pair.size_parent - 1);
  ParticleSet initial_distribution(N, A->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const double pos_orig = positions.at(dimx).at(px);
      initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] =
          velocities.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<INT>("ID")][px][0] = rank;
    initial_distribution[Sym<INT>("ID")][px][1] = px;
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = rank;
  }
  A->add_particles_local(initial_distribution);

  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  auto lambda_advect = [&] {
    auto t0 = profile_timestamp();
    particle_loop(
        A,
        [=](auto P, auto V) {
          for (int dimx = 0; dimx < ndim; dimx++) {
            P.at(dimx) += dt * V.at(dimx);
          }
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
        ->execute();
    sycl_target->profile_map.inc("Advect", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };

  auto lambda_check_owning_cell = [&] {
    Array<OneD, NekDouble> global_coord(3);
    Array<OneD, NekDouble> xi(3);
    Array<OneD, NekDouble> eta(3);
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A->position_dat->cell_dat.get_cell(cellx);
      auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
      auto reference_positions =
          A->get_cell(Sym<REAL>("NESO_REFERENCE_POSITIONS"), cellx);

      const int cell_nektar = cell_id_translation.map_to_nektar[cellx];
      auto geom = geoms_3d[cell_nektar];

      int shape_type = geom->GetShapeType();
      Newton::XMapNewton<Newton::MappingGeneric3D> mapper(sycl_target, geom);

      for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {
        const int cell_neso = (*cell_ids)[0][rowx];
        ASSERT_EQ(cell_neso, cellx);

        xi[0] = reference_positions->at(rowx, 0);
        xi[1] = reference_positions->at(rowx, 1);
        xi[2] = reference_positions->at(rowx, 2);
        global_coord[0] = geom->GetCoord(0, xi);
        global_coord[1] = geom->GetCoord(1, xi);
        global_coord[2] = geom->GetCoord(2, xi);

        // check the global coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double err_abs =
              ABS(positions->at(rowx, dimx) - global_coord[dimx]);
          const double err_rel =
              err_abs > 0.0 ? err_abs / std::abs(global_coord[dimx]) : err_abs;
          ASSERT_TRUE((err_abs <= tol) || (err_rel <= tol));
          ASSERT_TRUE(std::fabs((double)eta[dimx]) < (1.0 + tol));
        }
      }
    }
  };

  auto boundary_loop = particle_loop(
      "boundary_loop", A,
      [=](auto P) {
        REAL x = P.at(0);
        REAL y = P.at(1);
        REAL z = P.at(2);
        REAL r, phi, theta;
        ExternalCommon::cartesian_to_torus_cylindrical(torus_radius_major, x, y,
                                                       z, &r, &phi, &theta);

        const REAL test_pi = 3.14159265358979323846;
        phi = Kernel::fmod(phi + 2.0 * test_pi, 0.5 * test_pi);
        r = Kernel::fmod(r, torus_radius_minor);

        ExternalCommon::torus_cylindrical_to_cartesian(torus_radius_major, r,
                                                       phi, theta, &x, &y, &z);
        P.at(0) = x;
        P.at(1) = y;
        P.at(2) = z;
      },
      Access::write(Sym<REAL>("P")));

  REAL T = 0.0;
  for (int stepx = 0; stepx < Nsteps; stepx++) {
    boundary_loop->execute();
    A->hybrid_move();
    cell_id_translation.execute();
    A->cell_move();
    lambda_check_owning_cell();
    lambda_advect();
    T += dt;
  }

  mesh->free();
}
