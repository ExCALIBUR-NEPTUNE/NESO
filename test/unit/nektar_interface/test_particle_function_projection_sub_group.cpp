#include "nektar_interface/function_projection.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include "test_helper_utilities.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/DisContField.h>

namespace {
template <typename FIELD_TYPE>
static inline void wrapper(const std::string mesh_file,
                           const std::string conditions_file) {
  const int N_total = 20000;

  TestUtilities::TestResourceSession resource_session(mesh_file,
                                                      conditions_file);
  auto session = resource_session.session;
  auto graph = SpatialDomains::MeshGraph::Read(session);

  auto field = std::make_shared<ContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = mesh->get_ndim();
  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("ID"), 1),
      ParticleProp(Sym<REAL>("FUNC_EVALS"), 2),
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
      initial_distribution[Sym<REAL>("FUNC_EVALS")][px][0] = 2.0;
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

  auto Aeven = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));
  auto Aodd = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 1; },
      Access::read(Sym<INT>("ID")));

  auto field_project =
      std::make_shared<FieldProject<ContField>>(field, A, cell_id_translation);
  field_project->testing_enable();

  const int ncoeffs = field->GetNcoeffs();
  std::vector<REAL> to_test(ncoeffs);
  auto lambda_reset_to_test = [&]() {
    for (int cx = 0; cx < ncoeffs; cx++) {
      to_test.at(cx) = 0.0;
    }
  };
  auto lambda_add_to_test = [&]() {
    double *rhs_host;
    double *rhs_device;
    field_project->testing_get_rhs(&rhs_host, &rhs_device);
    for (int cx = 0; cx < ncoeffs; cx++) {
      to_test.at(cx) += rhs_device[cx];
    }
  };

  auto lambda_compare_with_to_test = [&]() {
    double *rhs_host;
    double *rhs_device;
    field_project->testing_get_rhs(&rhs_host, &rhs_device);
    for (int cx = 0; cx < ncoeffs; cx++) {
      const auto a = rhs_device[cx];
      const auto b = to_test.at(cx);
      const double err_abs = std::abs(a - b);
      const double err_rel =
          std::abs(a) > 0.0 ? err_abs / std::abs(a) : err_abs;
      const bool cond = err_abs < 1.0e-13 || err_rel < 1.0e-13;
      if (!cond) {
        nprint(cx, err_abs, err_rel);
      }
      ASSERT_TRUE(cond);
    }
  };

  auto lambda_set_weights = [&](auto aa, const REAL value) {
    particle_loop(
        aa, [=](auto FUNC_EVALS) { FUNC_EVALS.at(1) = value; },
        Access::write(Sym<REAL>("FUNC_EVALS")))
        ->execute();
  };

  auto lambda_proj_all = [&]() {
    field_project->project(A, std::vector<Sym<REAL>>{Sym<REAL>("FUNC_EVALS")},
                           std::vector<int>{1});
  };
  auto lambda_proj_even = [&]() {
    field_project->project(Aeven,
                           std::vector<Sym<REAL>>{Sym<REAL>("FUNC_EVALS")},
                           std::vector<int>{1});
  };
  auto lambda_proj_odd = [&]() {
    field_project->project(Aodd,
                           std::vector<Sym<REAL>>{Sym<REAL>("FUNC_EVALS")},
                           std::vector<int>{1});
  };

  // project both and compare
  lambda_set_weights(Aeven, 2.224);
  lambda_set_weights(Aodd, 2.224);
  lambda_reset_to_test();

  lambda_proj_even();
  lambda_add_to_test();
  lambda_proj_odd();
  lambda_add_to_test();

  lambda_proj_all();
  lambda_compare_with_to_test();

  // project even and compare
  lambda_set_weights(Aeven, 2.224);
  lambda_set_weights(Aodd, 2.224);
  lambda_reset_to_test();

  lambda_proj_even();
  lambda_add_to_test();

  lambda_set_weights(Aodd, 0.0);
  lambda_proj_all();

  lambda_compare_with_to_test();

  // project odd and compare
  lambda_set_weights(Aeven, 2.224);
  lambda_set_weights(Aodd, 2.224);
  lambda_reset_to_test();

  lambda_proj_odd();
  lambda_add_to_test();

  lambda_set_weights(Aeven, 0.0);
  lambda_proj_all();

  lambda_compare_with_to_test();

  A->free();
  sycl_target->free();
  mesh->free();
}
} // namespace

TEST(ParticleFunctionProjectionSubGroup, 2D) {
  wrapper<MultiRegions::ContField>("square_triangles_quads_nummodes_6.xml",
                                   "conditions_cg.xml");
}
TEST(ParticleFunctionProjectionSubGroup, 3DContField) {
  wrapper<MultiRegions::ContField>(
      "reference_all_types_cube/conditions_cg.xml",
      "reference_all_types_cube/linear_non_regular_0.5.xml");
}
TEST(ParticleFunctionProjectionSubGroup, 3DDisContFieldHex) {
  wrapper<MultiRegions::DisContField>("reference_hex_cube/conditions.xml",
                                      "reference_hex_cube/hex_cube_0.5.xml");
}
TEST(ParticleFunctionProjectionSubGroup, 3DDisContFieldPrismTet) {
  wrapper<MultiRegions::DisContField>(
      "reference_prism_tet_cube/conditions.xml",
      "reference_prism_tet_cube/prism_tet_cube_0.5.xml");
}
