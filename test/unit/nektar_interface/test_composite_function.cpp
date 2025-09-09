#include <SpatialDomains/MeshGraphIO.h>

#include "../../unit/nektar_interface/test_helper_utilities.hpp"

using namespace CompositeInteraction;

TEST(CompositeInteraction, SurfaceFunction3DInit) {

  const std::string filename_conditions =
      "reference_all_types_cube/conditions.xml";
  const std::string filename_mesh =
      "reference_all_types_cube/linear_non_regular_0.5.xml";
  const int ndim = 3;

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {100, 200, 300};
  boundary_groups[1] = {400, 500, 600};

  auto prototype_function = std::make_shared<DisContField>(session, graph, "u");

  auto composite_function_context = std::make_shared<CompositeFunctionContext>(
      sycl_target, graph, prototype_function, boundary_groups);

  auto func0 = composite_function_context->create_function(0);
  auto func1 = composite_function_context->create_function(1);

  ASSERT_EQ(func0->exp_lists.size(), 3);
  ASSERT_EQ(func1->exp_lists.size(), 3);

  auto lambda_get_geoms = [&](auto &func) -> std::vector<INT> {
    std::vector<INT> tmp_geoms;

    for (auto &exp_list : func->exp_lists) {
      if (exp_list != nullptr) {
        const auto exp_list_size = exp_list->GetExpSize();
        for (int ex = 0; ex < exp_list_size; ex++) {
          auto geom = exp_list->GetExp(ex)->GetGeom();
          const int geom_id = geom->GetGlobalID();
          tmp_geoms.push_back(geom_id);
        }
      }
    }

    return tmp_geoms;
  };

  ASSERT_EQ(lambda_get_geoms(func0),
            composite_function_context->get_owned_geoms(0));
  ASSERT_EQ(lambda_get_geoms(func1),
            composite_function_context->get_owned_geoms(1));

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto composite_intersection = std::make_shared<CompositeIntersection>(
      sycl_target, mesh, boundary_groups, prototype_function);

  auto funca = composite_intersection->create_function(0);
  auto funcb = composite_intersection->create_function(1);

  ASSERT_EQ(lambda_get_geoms(func0), lambda_get_geoms(funca));
  ASSERT_EQ(lambda_get_geoms(func1), lambda_get_geoms(funcb));

  composite_intersection->free();
  sycl_target->free();
  mesh->free();
}

TEST(CompositeInteraction, SurfaceFunction3DProjEval) {

  const std::string filename_conditions =
      "reference_all_types_cube/conditions.xml";
  const std::string filename_mesh =
      "reference_all_types_cube/linear_non_regular_0.5.xml";
  const int ndim = 3;

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {100, 200, 300};
  boundary_groups[1] = {400, 500, 600};

  auto prototype_function = std::make_shared<DisContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto composite_intersection = std::make_shared<CompositeIntersection>(
      sycl_target, mesh, boundary_groups, prototype_function);

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int cell_count = domain->mesh->get_cell_count();
  const int npart_per_cell = 2;

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int N = cell_count * npart_per_cell;
  ParticleSet initial_distribution(N, A->get_particle_spec());

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;

  const int rank = sycl_target->comm_pair.rank_parent;
  std::mt19937 rng(12234234 + rank);

  uniform_within_elements(graph, npart_per_cell, positions, cells, 1.0e-12,
                          rng);
  std::uniform_real_distribution<> dist(-2.0, 2.0);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const double pos_orig = positions[dimx][px];
      initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      initial_distribution[Sym<REAL>("V")][px][dimx] = dist(rng);
    }

    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }
  A->add_particles_local(initial_distribution);

  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  particle_loop(
      A,
      [=](auto V) {
        for (int dx = 0; dx < ndim; dx++) {
          const REAL v = V.at(dx);
          if (Kernel::abs(v) < 0.1) {
            V.at(dx) = (v < 0.0) ? -1.0 : 1.0;
          }
        }
      },
      Access::write(Sym<REAL>("V")))
      ->execute();

  composite_intersection->pre_integration(A);
  particle_loop(
      A,
      [=](auto P, auto V) {
        REAL vv = 0.0;

        for (int dx = 0; dx < ndim; dx++) {
          vv += V.at(dx) * V.at(dx);
        }
        const REAL iv = 1.0 / Kernel::sqrt(vv);

        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) += 1000.0 * V.at(dx) * iv;
        }
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
      ->execute();
  auto groups = composite_intersection->get_intersections(A);

  A->add_particle_dat(Sym<INT>("PD_NESO_PARTICLES_BOUNDARY_METADATA"), 2);
  A->add_particle_dat(Sym<REAL>("PD_NESO_BOUNDARY_REFERENCE_POSITIONS"),
                      ndim - 1);

  particle_loop(
      A, [=](auto X) { X.at(0) = -1; },
      Access::write(Sym<INT>("PD_NESO_PARTICLES_BOUNDARY_METADATA")))
      ->execute();

  for (auto groupx : groups) {
    copy_ephemeral_dat_to_particle_dat(
        groupx.second, Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA"),
        Sym<INT>("PD_NESO_PARTICLES_BOUNDARY_METADATA"));
    copy_ephemeral_dat_to_particle_dat(
        groupx.second, Sym<REAL>("NESO_BOUNDARY_REFERENCE_POSITIONS"),
        Sym<REAL>("PD_NESO_BOUNDARY_REFERENCE_POSITIONS"));
  }

  auto func0 = composite_intersection->create_function(0);
  auto func1 = composite_intersection->create_function(1);

  const int dof_seed = 12241234;
  std::uniform_real_distribution<> dof_dist(-1.0, 1.0);

  auto lambda_init_funcs = [&](auto func) {
    auto h_dofs = func->get_dofs();
    const std::size_t num_exp_lists = func->exp_lists.size();

    for (std::size_t ex = 0; ex < num_exp_lists; ex++) {
      auto exp_list = func->exp_lists.at(ex);
      const int num_expansions = exp_list->GetExpSize();
      for (int fx = 0; fx < num_expansions; fx++) {
        auto exp = exp_list->GetExp(fx);
        std::mt19937 dof_rng(dof_seed + exp->GetGeom()->GetGlobalID());
        const int num_dofs = exp->GetNcoeffs();
        ASSERT_EQ(num_dofs, h_dofs[ex][fx].size());
        for (int dx = 0; dx < num_dofs; dx++) {
          h_dofs[ex][fx][dx] = dof_dist(dof_rng);
        }
      }
    }
    func->set_dofs(h_dofs);

    auto h_dofs2 = func->get_dofs();

    ASSERT_EQ(h_dofs, h_dofs2);
  };

  lambda_init_funcs(func0);
  lambda_init_funcs(func1);

  A->print(Sym<REAL>("Q"));

  composite_intersection->function_evaluate(groups.at(0), Sym<REAL>("Q"), 0,
                                            false, func0);
  composite_intersection->function_evaluate(groups.at(1), Sym<REAL>("Q"), 0,
                                            false, func1);

  A->print(Sym<REAL>("Q"));

  composite_intersection->free();
  sycl_target->free();
  mesh->free();
}
