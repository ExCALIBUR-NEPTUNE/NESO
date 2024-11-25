#include "nektar_interface/function_evaluation.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include "test_helper_utilities.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/DisContField.h>

TEST(ParticleFunctionEvaluationSubGroup, 2D) {
  const int N_total = 20000;

  TestUtilities::TestResourceSession resource_session(
      "square_triangles_quads_nummodes_6.xml", "conditions_cg.xml");
  auto session = resource_session.session;
  auto graph = SpatialDomains::MeshGraph::Read(session);

  auto cont_field = std::make_shared<ContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("ID"), 1),
      ParticleProp(Sym<REAL>("FUNC_EVALS"), 1),
      ParticleProp(Sym<REAL>("TEST_FUNC_EVALS"), 1),
  };

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain->mesh, A->position_dat, A->cell_id_dat,
      A->mpi_rank_dat);

  pbc.execute();
  mesh_hierarchy_global_map.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  auto lambda_f = [&](const NekDouble x, const NekDouble y) {
    return 2.0 * (x + 0.5) * (x - 0.5) * (y + 0.8) * (y - 0.8);
  };

  interpolate_onto_nektar_field_2d(lambda_f, cont_field);

  // create evaluation object
  auto field_evaluate = std::make_shared<FieldEvaluate<ContField>>(
      cont_field, A, cell_id_translation);

  // evaluate field at particle locations
  field_evaluate->evaluate(Sym<REAL>("FUNC_EVALS"));
  particle_loop(
      A,
      [=](auto FUNC_EVALS, auto TEST_FUNC_EVALS) {
        TEST_FUNC_EVALS.at(0) = FUNC_EVALS.at(0);
        FUNC_EVALS.at(0) = -1.0;
        ;
      },
      Access::write(Sym<REAL>("FUNC_EVALS")),
      Access::write(Sym<REAL>("TEST_FUNC_EVALS")))
      ->execute();

  auto Aeven = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));
  auto Aodd = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 1; },
      Access::read(Sym<INT>("ID")));

  auto loop_reset = particle_loop(
      A, [=](auto FUNC_EVALS) { FUNC_EVALS.at(0) = -1.0; },
      Access::write(Sym<REAL>("FUNC_EVALS")));

  auto lambda_check = [&](const int comp) {
    ErrorPropagate ep(sycl_target);
    auto k_ep = ep.device_ptr();
    particle_loop(
        A,
        [=](auto ID, auto FUNC_EVALS, auto TEST_FUNC_EVALS) {
          if (ID.at(0) % 2 == comp) {
            NESO_KERNEL_ASSERT(
                Kernel::abs(FUNC_EVALS.at(0) - TEST_FUNC_EVALS.at(0)) < 1.0e-15,
                k_ep);
          } else {
            NESO_KERNEL_ASSERT(FUNC_EVALS.at(0) == -1.0, k_ep);
          }
        },
        Access::read(Sym<INT>("ID")), Access::read(Sym<REAL>("FUNC_EVALS")),
        Access::read(Sym<REAL>("TEST_FUNC_EVALS")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  };

  loop_reset->execute();
  field_evaluate->evaluate(Aeven, Sym<REAL>("FUNC_EVALS"));
  lambda_check(0);
  loop_reset->execute();
  field_evaluate->evaluate(Aodd, Sym<REAL>("FUNC_EVALS"));
  lambda_check(1);

  A->free();
  sycl_target->free();
  mesh->free();
}

template <typename FIELD_TYPE>
static inline void evaluation_wrapper_3d(std::string condtions_file_s,
                                         std::string mesh_file_s,
                                         const double tol) {

  const int N_total = 16000;
  TestUtilities::TestResourceSession resource_session(mesh_file_s,
                                                      condtions_file_s);
  auto session = resource_session.session;
  auto graph = SpatialDomains::MeshGraph::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  std::mt19937 rng{182348};

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 3;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("FUNC_EVALS"), 1),
                             ParticleProp(Sym<REAL>("TEST_FUNC_EVALS"), 1),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);
  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;
  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");
  const int cell_count = domain->mesh->get_cell_count();
  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }

      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  pbc.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  auto field = std::make_shared<FIELD_TYPE>(session, graph, "u");

  auto lambda_f = [&](const NekDouble x, const NekDouble y, const NekDouble z) {
    return std::pow((x + 1.0) * (x - 1.0) * (y + 1.0) * (y - 1.0) * (z + 1.0) *
                        (z - 1.0),
                    4);
  };
  interpolate_onto_nektar_field_3d(lambda_f, field);
  NESOCellsToNektarExp map_cells_to_exp(field, cell_id_translation);

  auto field_evaluate = std::make_shared<FieldEvaluate<FIELD_TYPE>>(
      field, A, cell_id_translation);

  // evaluate field at particle locations
  field_evaluate->evaluate(Sym<REAL>("FUNC_EVALS"));
  particle_loop(
      A,
      [=](auto FUNC_EVALS, auto TEST_FUNC_EVALS) {
        TEST_FUNC_EVALS.at(0) = FUNC_EVALS.at(0);
        FUNC_EVALS.at(0) = -1.0;
        ;
      },
      Access::write(Sym<REAL>("FUNC_EVALS")),
      Access::write(Sym<REAL>("TEST_FUNC_EVALS")))
      ->execute();

  auto Aeven = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));
  auto Aodd = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 1; },
      Access::read(Sym<INT>("ID")));

  auto loop_reset = particle_loop(
      A, [=](auto FUNC_EVALS) { FUNC_EVALS.at(0) = -1.0; },
      Access::write(Sym<REAL>("FUNC_EVALS")));

  auto lambda_check = [&](const int comp) {
    ErrorPropagate ep(sycl_target);
    auto k_ep = ep.device_ptr();
    particle_loop(
        A,
        [=](auto ID, auto FUNC_EVALS, auto TEST_FUNC_EVALS) {
          if (ID.at(0) % 2 == comp) {
            NESO_KERNEL_ASSERT(
                Kernel::abs(FUNC_EVALS.at(0) - TEST_FUNC_EVALS.at(0)) < 1.0e-15,
                k_ep);
          } else {
            NESO_KERNEL_ASSERT(FUNC_EVALS.at(0) == -1.0, k_ep);
          }
        },
        Access::read(Sym<INT>("ID")), Access::read(Sym<REAL>("FUNC_EVALS")),
        Access::read(Sym<REAL>("TEST_FUNC_EVALS")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  };

  loop_reset->execute();
  field_evaluate->evaluate(Aeven, Sym<REAL>("FUNC_EVALS"));
  lambda_check(0);
  loop_reset->execute();
  field_evaluate->evaluate(Aodd, Sym<REAL>("FUNC_EVALS"));
  lambda_check(1);

  A->free();
  mesh->free();
}

TEST(ParticleFunctionEvaluationSubGroup, 3DContField) {
  evaluation_wrapper_3d<MultiRegions::ContField>(
      "reference_all_types_cube/conditions_cg.xml",
      "reference_all_types_cube/linear_non_regular_0.5.xml", 1.0e-15);
}
TEST(ParticleFunctionEvaluationSubGroup, 3DDisContFieldHex) {
  evaluation_wrapper_3d<MultiRegions::DisContField>(
      "reference_hex_cube/conditions.xml",
      "reference_hex_cube/hex_cube_0.5.xml", 1.0e-15);
}
TEST(ParticleFunctionEvaluationSubGroup, 3DDisContFieldPrismTet) {
  evaluation_wrapper_3d<MultiRegions::DisContField>(
      "reference_prism_tet_cube/conditions.xml",
      "reference_prism_tet_cube/prism_tet_cube_0.5.xml", 1.0e-15);
}
