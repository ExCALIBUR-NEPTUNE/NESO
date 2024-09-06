#include "../../unit/nektar_interface/test_helper_utilities.hpp"

TEST(CompositeInteraction, MASTUReflection) {

  const int npart_per_cell = 10;
  const REAL dt = 0.05;
  const int N_steps = 500;
  const int ndim = 2;

  TestUtilities::TestResourceSession resources_session(
      "MASTU-2D/mastu_cd.xml", "MASTU-2D/conditions_nummodes_2.xml");
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraph::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());
  auto config = std::make_shared<ParameterStore>();
  config->set<REAL>("MapParticles2DRegular/tol", 1.0e-8);
  config->set<REAL>("CompositeIntersection/newton_tol", 1.0e-10);
  config->set<REAL>("CompositeIntersection/line_intersection_tol", 1.0e-10);
  config->set<REAL>("NektarCompositeTruncatedReflection/reset_distance",
                    1.0e-6);

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh, config);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<REAL>("TSP"), 2),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;
  std::mt19937 rng(534234 + rank);

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  rng = uniform_within_elements(graph, npart_per_cell, positions, cells,
                                1.0e-10, rng);

  const int N = cells.size();
  auto velocities = Particles::normal_distribution(N, 2, 0.0, 500.0, rng);

  int id_offset = 0;
  MPICHK(MPI_Exscan(&N, &id_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

  if (N > 0) {
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        initial_distribution[Sym<REAL>("P")][px][dimx] =
            positions.at(dimx).at(px);
        initial_distribution[Sym<REAL>("V")][px][dimx] =
            velocities.at(dimx).at(px);
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
      initial_distribution[Sym<INT>("ID")][px][0] = id_offset + px;
    }
    A->add_particles_local(initial_distribution);
  }
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  H5Part h5part("MASTU_reflection.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"),
                Sym<INT>("ID"));

  std::vector<int> reflection_composites = {100, 101, 102, 103, 104,
                                            105, 106, 107, 108};
  auto reflection = std::make_shared<NektarCompositeTruncatedReflection>(
      Sym<REAL>("V"), Sym<REAL>("TSP"), sycl_target, mesh,
      reflection_composites, config);

  auto lambda_apply_timestep_reset = [&](auto aa) {
    particle_loop(
        aa,
        [=](auto TSP) {
          TSP.at(0) = 0.0;
          TSP.at(1) = 0.0;
        },
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };

  auto lambda_apply_advection_step =
      [&](ParticleSubGroupSharedPtr iteration_set) -> void {
    particle_loop(
        "euler_advection", iteration_set,
        [=](auto V, auto P, auto TSP) {
          const REAL dt_left = dt - TSP.at(0);
          if (dt_left > 0.0) {
            for (int dx = 0; dx < ndim; dx++) {
              P.at(dx) += dt_left * V.at(dx);
            }
            TSP.at(0) = dt;
            TSP.at(1) = dt_left;
          }
        },
        Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("P")),
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };

  auto lambda_pre_advection = [&](auto aa) { reflection->pre_advection(aa); };

  auto lambda_apply_boundary_conditions = [&](auto aa) {
    reflection->execute(aa);
  };

  auto lambda_find_partial_moves = [&](auto aa) {
    return static_particle_sub_group(
        A, [=](auto TSP) { return TSP.at(0) < dt; },
        Access::read(Sym<REAL>("TSP")));
  };

  auto lambda_partial_moves_remaining = [&](auto aa) -> bool {
    const int size = aa->get_npart_local();
    int size_global;
    MPICHK(MPI_Allreduce(&size, &size_global, 1, MPI_INT, MPI_SUM,
                         sycl_target->comm_pair.comm_parent));
    return size_global > 0;
  };

  auto lambda_apply_timestep = [&](auto aa) {
    lambda_apply_timestep_reset(aa);
    lambda_pre_advection(aa);
    lambda_apply_advection_step(aa);
    lambda_apply_boundary_conditions(aa);
    aa = lambda_find_partial_moves(aa);
    while (lambda_partial_moves_remaining(aa)) {
      lambda_pre_advection(aa);
      lambda_apply_advection_step(aa);
      lambda_apply_boundary_conditions(aa);
      aa = lambda_find_partial_moves(aa);
    }
  };

  lambda_pre_advection(particle_sub_group(A));
  lambda_apply_boundary_conditions(particle_sub_group(A));
  h5part.write();
  h5part.close();
  for (int stepx = 0; stepx < N_steps; stepx++) {
    lambda_apply_timestep(static_particle_sub_group(A));
    A->hybrid_move();
    cell_id_translation->execute();
    A->cell_move();

    h5part.write();
    h5part.close();
    if (!rank) {
      nprint(stepx);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
