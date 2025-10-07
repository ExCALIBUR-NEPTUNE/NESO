#include <SpatialDomains/MeshGraphIO.h>

#include "../../unit/nektar_interface/test_helper_utilities.hpp"

using namespace CompositeInteraction;

class CompositeInteractionSurfaceFunctionAllD
    : public testing::TestWithParam<std::tuple<std::string, std::string, int>> {
};

TEST_P(CompositeInteractionSurfaceFunctionAllD, SurfaceFunctionInit) {
  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {100, 200, 300};
  boundary_groups[1] = {400};
  if (ndim > 2) {
    boundary_groups[1] = {400, 500, 600};
  }

  auto prototype_function = std::make_shared<DisContField>(session, graph, "u");

  auto composite_function_context = std::make_shared<CompositeFunctionContext>(
      sycl_target, graph, prototype_function, boundary_groups);

  auto func0 = composite_function_context->create_function(0);
  auto func1 = composite_function_context->create_function(1);

  ASSERT_EQ(func0->exp_lists.size(), boundary_groups.at(0).size());
  ASSERT_EQ(func1->exp_lists.size(), boundary_groups.at(1).size());

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

TEST_P(CompositeInteractionSurfaceFunctionAllD, SurfaceFunctionEval) {

  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {100, 200, 300};
  boundary_groups[1] = {400};
  if (ndim > 2) {
    boundary_groups[1] = {400, 500, 600};
  }

  auto prototype_function = std::make_shared<DisContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto composite_intersection = std::make_shared<CompositeIntersection>(
      sycl_target, mesh, boundary_groups, prototype_function);

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int cell_count = domain->mesh->get_cell_count();
  const int npart_per_cell = 8;

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("ID"), 2)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int N = cell_count * npart_per_cell;
  ParticleSet initial_distribution(N, A->get_particle_spec());

  std::vector<int> cells;

  const int rank = sycl_target->comm_pair.rank_parent;
  std::mt19937 rng(12234234 + rank);

  double extents[3] = {2, 2, 2};
  auto positions = uniform_within_extents(N, ndim, extents, rng);

  std::uniform_real_distribution<> dist(-2.0, 2.0);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const double pos_orig = positions[dimx][px];
      initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig - 1.0;
      initial_distribution[Sym<REAL>("V")][px][dimx] = dist(rng);
    }

    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = rank;
    initial_distribution[Sym<INT>("ID")][px][1] = px;
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

  auto lambda_check_num_modes = [&](const auto shape_type) {
    int num_modes = composite_intersection->composite_function_context
                        ->map_shape_type_to_num_modes.at(shape_type);

    int num_modes_check = num_modes;
    MPICHK(MPI_Bcast(&num_modes_check, 1, MPI_INT, 0, MPI_COMM_WORLD));
    ASSERT_EQ(num_modes_check, num_modes);

    num_modes = composite_intersection->composite_function_context
                    ->map_shape_type_to_total_num_modes.at(shape_type)
                    .at(0);
    num_modes_check = num_modes;
    MPICHK(MPI_Bcast(&num_modes_check, 1, MPI_INT, 0, MPI_COMM_WORLD));
    ASSERT_EQ(num_modes_check, num_modes);

    num_modes = composite_intersection->composite_function_context
                    ->map_shape_type_to_total_num_modes.at(shape_type)
                    .at(1);
    num_modes_check = num_modes;
    MPICHK(MPI_Bcast(&num_modes_check, 1, MPI_INT, 0, MPI_COMM_WORLD));
    ASSERT_EQ(num_modes_check, num_modes);
  };

  if (ndim == 3) {
    lambda_check_num_modes(eQuadrilateral);
    lambda_check_num_modes(eTriangle);
  } else {
    lambda_check_num_modes(eSegment);
  }

  const int dof_seed = 12241234;
  std::uniform_real_distribution<> dof_dist(-1.0, 1.0);

  auto lambda_make_dofs = [&](auto geom_id,
                              const int num_dofs) -> std::vector<REAL> {
    std::mt19937 dof_rng(dof_seed + geom_id);
    std::vector<REAL> dofs(num_dofs);
    for (int dx = 0; dx < num_dofs; dx++) {
      dofs[dx] = dof_dist(dof_rng);
    }

    return dofs;
  };

  auto lambda_init_funcs = [&](auto func) {
    auto h_dofs = func->get_dofs();
    const std::size_t num_exp_lists = func->exp_lists.size();

    for (std::size_t ex = 0; ex < num_exp_lists; ex++) {
      auto exp_list = func->exp_lists.at(ex);
      if (exp_list) {
        const int num_expansions = exp_list->GetExpSize();
        for (int fx = 0; fx < num_expansions; fx++) {
          auto exp = exp_list->GetExp(fx);
          const int geom_id = exp->GetGeom()->GetGlobalID();
          const int num_dofs = exp->GetNcoeffs();
          ASSERT_EQ(num_dofs, h_dofs[ex][fx].size());
          auto new_dofs = lambda_make_dofs(geom_id, num_dofs);
          for (int dx = 0; dx < num_dofs; dx++) {
            h_dofs[ex][fx][dx] = new_dofs[dx];
          }
        }
      }
    }
    func->set_dofs(h_dofs);

    auto h_dofs2 = func->get_dofs();

    ASSERT_EQ(h_dofs, h_dofs2);
  };

  lambda_init_funcs(func0);
  lambda_init_funcs(func1);

  composite_intersection->function_evaluate(groups.at(0), Sym<REAL>("Q"), 0,
                                            false, func0);
  composite_intersection->function_evaluate(groups.at(1), Sym<REAL>("Q"), 0,
                                            false, func1);

  std::map<int, CompositeFunctionSharedPtr> map_group_to_func;
  map_group_to_func[0] = func0;
  map_group_to_func[1] = func1;

  std::map<int, std::vector<REAL>> map_group_to_stage_dofs;
  map_group_to_stage_dofs[0] = func0->get_stage_dofs_linear();
  map_group_to_stage_dofs[1] = func1->get_stage_dofs_linear();

  std::map<int, std::set<INT>> map_group_to_extend_geoms;
  map_group_to_extend_geoms[0] =
      composite_intersection->get_boundary_mesh_interface(0)
          ->get_extended_pattern_geom_ids();
  map_group_to_extend_geoms[1] =
      composite_intersection->get_boundary_mesh_interface(1)
          ->get_extended_pattern_geom_ids();

  const int max_num_dofs =
      composite_intersection->composite_function_context->max_num_dofs;

  auto lambda_check_eval_dofs = [&](auto func) {
    const int group = func->boundary_group;
    auto boundary_mesh_interface =
        composite_intersection->get_boundary_mesh_interface(group);

    std::set<INT> pattern_geom_ids =
        boundary_mesh_interface->get_extended_pattern_geom_ids();

    std::set<INT> linear_offsets;

    for (INT geom_id : pattern_geom_ids) {
      auto geom = composite_intersection->composite_collections
                      ->map_geom_id_to_geoms.at(geom_id);
      const auto shape_type = geom->GetShapeType();
      const int num_modes = composite_intersection->composite_function_context
                                ->map_shape_type_to_num_modes.at(shape_type);
      const int num_dofs =
          BasisReference::get_total_num_modes(shape_type, num_modes);

      auto linear_geom_index =
          boundary_mesh_interface->get_seq_index_from_geom_id(geom_id);
      ASSERT_EQ(linear_offsets.count(linear_geom_index), 0);
      linear_offsets.insert(linear_geom_index);

      auto h_correct_dofs = lambda_make_dofs(geom_id, num_dofs);
      auto h_to_test_dofs = std::vector<REAL>(num_dofs);
      for (int ix = 0; ix < num_dofs; ix++) {
        h_to_test_dofs[ix] =
            map_group_to_stage_dofs[group]
                                   [linear_geom_index * max_num_dofs + ix];
      }

      ASSERT_EQ(h_correct_dofs, h_to_test_dofs);
    }
  };

  lambda_check_eval_dofs(func0);
  lambda_check_eval_dofs(func1);

  std::vector<double> basis_evaluations;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto REF_POSITIONS =
        A->get_cell(Sym<REAL>("PD_NESO_BOUNDARY_REFERENCE_POSITIONS"), cellx);
    auto Q = A->get_cell(Sym<REAL>("Q"), cellx);
    auto METADATA =
        A->get_cell(Sym<INT>("PD_NESO_PARTICLES_BOUNDARY_METADATA"), cellx);
    auto ID = A->get_cell(Sym<INT>("ID"), cellx);
    const int nrow = METADATA->nrow;
    for (int rx = 0; rx < nrow; rx++) {
      const int group = static_cast<int>(METADATA->at(rx, 0));
      if (group != -1) {
        const int geom_id = static_cast<int>(METADATA->at(rx, 1));

        ASSERT_EQ(1, map_group_to_extend_geoms.at(group).count(geom_id));

        auto geom = composite_intersection->composite_collections
                        ->map_geom_id_to_geoms.at(geom_id);
        const auto shape_type = geom->GetShapeType();
        const int num_modes = composite_intersection->composite_function_context
                                  ->map_shape_type_to_num_modes.at(shape_type);
        const int num_dofs =
            BasisReference::get_total_num_modes(shape_type, num_modes);
        basis_evaluations.resize(num_dofs);

        if (ndim == 3) {
          const REAL xi0 = REF_POSITIONS->at(rx, 0);
          const REAL xi1 = REF_POSITIONS->at(rx, 1);
          REAL eta0 = -2.0;
          REAL eta1 = -2.0;
          GeometryInterface::loc_coord_to_loc_collapsed_2d(shape_type, xi0, xi1,
                                                           &eta0, &eta1);

          BasisReference::eval_modes(shape_type, num_modes, eta0, eta1, 0.0,
                                     basis_evaluations);

        } else {
          const REAL eta0 = REF_POSITIONS->at(rx, 0);
          BasisReference::eval_modes(shape_type, num_modes, eta0, 0.0, 0.0,
                                     basis_evaluations);
        }

        auto dofs = lambda_make_dofs(geom_id, num_dofs);
        REAL correct = 0.0;
        for (int dofx = 0; dofx < num_dofs; dofx++) {
          correct += basis_evaluations[dofx] * dofs[dofx];
        }

        const REAL to_test = Q->at(rx, 0);
        const REAL error = minimum_absrel_error(correct, to_test);
        ASSERT_TRUE(error < 1.0e-10);
      }
    }
  }

  composite_intersection->free();
  sycl_target->free();
  mesh->free();
}

TEST_P(CompositeInteractionSurfaceFunctionAllD, SurfaceFunctionProjRHS) {

  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {100, 200, 300};
  boundary_groups[1] = {400};
  if (ndim > 2) {
    boundary_groups[1] = {400, 500, 600};
  }

  auto prototype_function = std::make_shared<DisContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto composite_intersection = std::make_shared<CompositeIntersection>(
      sycl_target, mesh, boundary_groups, prototype_function);

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int cell_count = domain->mesh->get_cell_count();
  const int npart_per_cell = 8;

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("ID"), 2)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int N = cell_count * npart_per_cell;
  ParticleSet initial_distribution(N, A->get_particle_spec());

  std::vector<int> cells;

  const int rank = sycl_target->comm_pair.rank_parent;
  std::mt19937 rng(12234234 + rank);
  double extents[3] = {2, 2, 2};
  auto positions = uniform_within_extents(N, 3, extents, rng);

  std::uniform_real_distribution<> dist(-2.0, 2.0);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const double pos_orig = positions[dimx][px];
      initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig - 1.0;
      initial_distribution[Sym<REAL>("V")][px][dimx] = dist(rng);
    }

    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = rank;
    initial_distribution[Sym<INT>("ID")][px][1] = px;
    initial_distribution[Sym<REAL>("Q")][px][0] = 0.01 + std::abs(dist(rng));
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

  auto lambda_compute_project_rhs = [&](const int group, auto func) {
    composite_intersection->composite_function_context
        ->function_project_initialise(
            func, composite_intersection->get_boundary_mesh_interface(group));
    composite_intersection->function_project_contribute(
        groups.at(group), Sym<REAL>("Q"), 0, false, func);
  };

  lambda_compute_project_rhs(0, func0);
  lambda_compute_project_rhs(1, func1);

  std::map<int, std::vector<REAL>> map_gid_to_local_rhs_dofs;

  std::vector<double> basis_evaluations;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto REF_POSITIONS =
        A->get_cell(Sym<REAL>("PD_NESO_BOUNDARY_REFERENCE_POSITIONS"), cellx);
    auto Q = A->get_cell(Sym<REAL>("Q"), cellx);
    auto METADATA =
        A->get_cell(Sym<INT>("PD_NESO_PARTICLES_BOUNDARY_METADATA"), cellx);
    auto ID = A->get_cell(Sym<INT>("ID"), cellx);
    const int nrow = METADATA->nrow;
    for (int rx = 0; rx < nrow; rx++) {
      const int group = static_cast<int>(METADATA->at(rx, 0));
      if (group != -1) {
        const int geom_id = static_cast<int>(METADATA->at(rx, 1));

        auto geom = composite_intersection->composite_collections
                        ->map_geom_id_to_geoms.at(geom_id);
        const auto shape_type = geom->GetShapeType();
        const int num_modes = composite_intersection->composite_function_context
                                  ->map_shape_type_to_num_modes.at(shape_type);
        const int num_dofs =
            BasisReference::get_total_num_modes(shape_type, num_modes);
        basis_evaluations.resize(num_dofs);

        if (ndim == 3) {
          const REAL xi0 = REF_POSITIONS->at(rx, 0);
          const REAL xi1 = REF_POSITIONS->at(rx, 1);
          REAL eta0 = -2.0;
          REAL eta1 = -2.0;
          GeometryInterface::loc_coord_to_loc_collapsed_2d(shape_type, xi0, xi1,
                                                           &eta0, &eta1);

          BasisReference::eval_modes(shape_type, num_modes, eta0, eta1, 0.0,
                                     basis_evaluations);
        } else {
          const REAL eta0 = REF_POSITIONS->at(rx, 0);
          BasisReference::eval_modes(shape_type, num_modes, eta0, 0.0, 0.0,
                                     basis_evaluations);
        }

        if (!map_gid_to_local_rhs_dofs.count(geom_id)) {
          map_gid_to_local_rhs_dofs[geom_id].resize(num_dofs);
          std::fill(map_gid_to_local_rhs_dofs[geom_id].begin(),
                    map_gid_to_local_rhs_dofs[geom_id].end(), 0.0);
        }

        std::transform(
            map_gid_to_local_rhs_dofs[geom_id].begin(),
            map_gid_to_local_rhs_dofs[geom_id].end(), basis_evaluations.begin(),
            map_gid_to_local_rhs_dofs[geom_id].begin(),
            [&](auto ax, auto bx) { return ax + bx * Q->at(rx, 0); });
      }
    }
  }

  // Test the local RHS values match the contributions from each particle.
  auto lambda_test_rhs_local_contributions = [&](const int group, auto func) {
    auto boundary_mesh_interface =
        composite_intersection->get_boundary_mesh_interface(group);

    auto h_dofs_stage = func->get_stage_dofs_linear();
    auto hit_geoms = boundary_mesh_interface->get_extended_pattern_geom_ids();

    for (auto exp_list : func->exp_lists) {
      if (exp_list) {
        const int num_expansions = exp_list->GetExpSize();
        for (int ex = 0; ex < num_expansions; ex++) {
          const int geom_id = exp_list->GetExp(ex)->GetGeom()->GetGlobalID();

          if (hit_geoms.count(geom_id)) {
            const auto linear_index =
                boundary_mesh_interface->get_seq_index_from_geom_id(geom_id);
            const auto linear_offset = linear_index * func->max_num_dofs;
            const int num_dofs = exp_list->GetExp(ex)->GetNcoeffs();

            const auto correct_dofs = map_gid_to_local_rhs_dofs.at(geom_id);
            for (int dx = 0; dx < num_dofs; dx++) {
              const REAL correct = correct_dofs.at(dx);
              const REAL to_test = h_dofs_stage.at(linear_offset + dx);
              const REAL error = minimum_absrel_error(correct, to_test);
              ASSERT_TRUE(error < 1.0e-10);
            }
          }
        }
      }
    }
  };

  lambda_test_rhs_local_contributions(0, func0);
  lambda_test_rhs_local_contributions(1, func1);

  composite_intersection->composite_function_context
      ->function_project_finalise_reduce(
          func0, composite_intersection->get_boundary_mesh_interface(0));
  composite_intersection->function_project_finalise_reduce(func1);

  // Test the reduced RHS values match the contributions from each MPI rank.
  auto lambda_check_reduced_rhs = [&](const int group, auto func) {
    auto boundary_mesh_interface =
        composite_intersection->get_boundary_mesh_interface(group);
    auto h_dofs = func->get_dofs_linear();
    auto h_stage_dofs = func->get_stage_dofs_linear();

    std::set<int> owned_geoms;
    std::map<int, int> map_geom_id_offset;
    int index = 0;
    for (auto exp_list : func->exp_lists) {
      if (exp_list) {
        const int num_expansions = exp_list->GetExpSize();
        for (int ex = 0; ex < num_expansions; ex++) {
          const int geom_id = exp_list->GetExp(ex)->GetGeom()->GetGlobalID();
          owned_geoms.insert(geom_id);
          map_geom_id_offset[geom_id] = index * func->max_num_dofs;
          index++;
        }
      }
    }

    const int max_num_dofs = func->max_num_dofs;
    auto hit_geoms = boundary_mesh_interface->get_extended_pattern_geom_ids();
    auto all_hit_geoms = set_all_reduce_union(hit_geoms, MPI_COMM_WORLD);

    std::vector<double> h_zero_dofs(max_num_dofs);
    std::fill(h_zero_dofs.begin(), h_zero_dofs.end(), 0.0);

    std::vector<double> h_tmp_dofs(max_num_dofs);
    std::vector<double> h_tmp_reduce_dofs(max_num_dofs);

    for (auto gx : all_hit_geoms) {
      std::fill(h_tmp_dofs.begin(), h_tmp_dofs.end(), 0.0);
      std::fill(h_tmp_reduce_dofs.begin(), h_tmp_reduce_dofs.end(), 0.0);

      int gx_bcast = gx;
      MPICHK(MPI_Bcast(&gx_bcast, 1, MPI_INT, 0, MPI_COMM_WORLD));
      ASSERT_EQ(gx_bcast, gx);

      double *h_to_reduce = h_zero_dofs.data();
      if (hit_geoms.count(gx)) {
        const auto linear_index =
            boundary_mesh_interface->get_seq_index_from_geom_id(gx);
        const auto linear_offset = linear_index * max_num_dofs;
        std::copy(h_stage_dofs.begin() + linear_offset,
                  h_stage_dofs.begin() + linear_offset + max_num_dofs,
                  h_tmp_dofs.begin());
        h_to_reduce = h_tmp_dofs.data();
      }

      MPICHK(MPI_Allreduce(h_to_reduce, h_tmp_reduce_dofs.data(), max_num_dofs,
                           MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

      if (owned_geoms.count(gx)) {
        const auto linear_offset = map_geom_id_offset.at(gx);
        for (int ix = 0; ix < max_num_dofs; ix++) {
          ASSERT_NEAR(h_tmp_reduce_dofs.at(ix), h_dofs.at(linear_offset + ix),
                      1.0e-10);
        }
      }
    }

    // If no particles hit that geom then the RHS values should still all be
    // zero
    for (auto gx : owned_geoms) {
      if (all_hit_geoms.count(gx) == 0) {
        const auto linear_offset = map_geom_id_offset.at(gx);
        for (int ix = 0; ix < max_num_dofs; ix++) {
          ASSERT_NEAR(0.0, h_dofs.at(linear_offset + ix), 1.0e-10);
        }
      }
    }
  };

  lambda_check_reduced_rhs(0, func0);
  lambda_check_reduced_rhs(1, func1);

  composite_intersection->free();
  sycl_target->free();
  mesh->free();
}

TEST_P(CompositeInteractionSurfaceFunctionAllD, SurfaceFunctionProjMassSolve) {

  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {100, 200, 300};
  boundary_groups[1] = {400};
  if (ndim > 2) {
    boundary_groups[1] = {400, 500, 600};
  }

  auto prototype_function = std::make_shared<DisContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto composite_intersection = std::make_shared<CompositeIntersection>(
      sycl_target, mesh, boundary_groups, prototype_function);

  auto func0 = composite_intersection->create_function(0);
  auto func1 = composite_intersection->create_function(1);

  std::mt19937 rng(12234234 + sycl_target->comm_pair.rank_parent);
  std::uniform_real_distribution<> dist(-2.0, 2.0);

  auto lambda_test_mass_solve = [&](const int group, auto func) {
    auto h_dofs = func->get_dofs();

    std::vector<Array<OneD, NekDouble>> inarrays(func->exp_lists.size());
    std::vector<Array<OneD, NekDouble>> outarrays(func->exp_lists.size());

    const int num_exp_lists = func->exp_lists.size();

    for (int exp_listx = 0; exp_listx < num_exp_lists; exp_listx++) {
      auto exp_list = func->exp_lists.at(exp_listx);
      if (exp_list) {
        const int num_dofs_list = exp_list->GetNcoeffs();
        const int num_elements = exp_list->GetExpSize();
        Array<OneD, NekDouble> inarray(num_dofs_list);
        Array<OneD, NekDouble> outarray(num_dofs_list);

        int index = 0;
        for (int ex = 0; ex < num_elements; ex++) {
          auto exp = exp_list->GetExp(ex);
          const int num_dofs = exp->GetNcoeffs();
          for (int dx = 0; dx < num_dofs; dx++) {
            const REAL value = dist(rng);
            h_dofs[exp_listx][ex][dx] = value;
            inarray[index] = value;
            outarray[index] = 0.0;
            index++;
          }
        }

        inarrays[exp_listx] = inarray;
        outarrays[exp_listx] = outarray;
      }
    }
    func->set_dofs(h_dofs);

    // Do the mass matrix solve with the test rhs
    for (int exp_listx = 0; exp_listx < num_exp_lists; exp_listx++) {
      auto exp_list = func->exp_lists.at(exp_listx);
      if (exp_list) {
        exp_list->MultiplyByElmtInvMass(inarrays[exp_listx],
                                        outarrays[exp_listx]);
      }
    }

    // do the actual mass solve
    auto boundary_mesh_interface =
        composite_intersection->get_boundary_mesh_interface(group);
    composite_intersection->function_project_finalise_mass_solve(func);

    h_dofs = func->get_dofs();

    for (int exp_listx = 0; exp_listx < num_exp_lists; exp_listx++) {
      auto exp_list = func->exp_lists.at(exp_listx);
      if (exp_list) {
        const int num_dofs_list = exp_list->GetNcoeffs();
        const int num_elements = exp_list->GetExpSize();

        int index = 0;
        for (int ex = 0; ex < num_elements; ex++) {
          auto exp = exp_list->GetExp(ex);
          const int num_dofs = exp->GetNcoeffs();
          for (int dx = 0; dx < num_dofs; dx++) {
            const REAL to_test = h_dofs[exp_listx][ex][dx];
            const REAL correct = outarrays[exp_listx][index];
            index++;
            const REAL error = minimum_absrel_error(correct, to_test);
            ASSERT_TRUE(error < 1.0e-14);
          }
        }
      }
    }
  };

  lambda_test_mass_solve(0, func0);
  lambda_test_mass_solve(1, func1);

  composite_intersection->free();
  sycl_target->free();
  mesh->free();
}

TEST_P(CompositeInteractionSurfaceFunctionAllD, SurfaceFunctionProjIntegrate) {

  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {100, 200, 300};
  boundary_groups[1] = {400};
  if (ndim > 2) {
    boundary_groups[1] = {400, 500, 600};
  }
  auto prototype_function = std::make_shared<DisContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto composite_intersection = std::make_shared<CompositeIntersection>(
      sycl_target, mesh, boundary_groups, prototype_function);

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int cell_count = domain->mesh->get_cell_count();
  const int npart_per_cell = 8;

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("ID"), 2)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int N = cell_count * npart_per_cell;
  ParticleSet initial_distribution(N, A->get_particle_spec());

  std::vector<int> cells;

  const int rank = sycl_target->comm_pair.rank_parent;
  std::mt19937 rng(12234234 + rank);
  double extents[3] = {2, 2, 2};
  auto positions = uniform_within_extents(N, 3, extents, rng);

  std::uniform_real_distribution<> dist(-2.0, 2.0);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const double pos_orig = positions[dimx][px];
      initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig - 1.0;
      initial_distribution[Sym<REAL>("V")][px][dimx] = dist(rng);
    }

    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = rank;
    initial_distribution[Sym<INT>("ID")][px][1] = px;
    initial_distribution[Sym<REAL>("Q")][px][0] = 0.01 + std::abs(dist(rng));
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

  auto func0 = composite_intersection->create_function(0);
  auto func1 = composite_intersection->create_function(1);

  auto lambda_compute_project_rhs = [&](const int group, auto func) -> REAL {
    composite_intersection->function_project(groups.at(group), Sym<REAL>("Q"),
                                             0, false, func);

    auto ga = std::make_shared<GlobalArray<REAL>>(sycl_target, 1);
    ga->fill(0.0);
    particle_loop(
        groups.at(group), [=](auto Q, auto GA) { GA.add(0, Q.at(0)); },
        Access::read(Sym<REAL>("Q")), Access::add(ga))
        ->execute();

    return ga->get().at(0);
  };

  const REAL mass_correct_0 = lambda_compute_project_rhs(0, func0);
  const REAL mass_correct_1 = lambda_compute_project_rhs(1, func1);

  const REAL mass_to_test_0 = CompositeInteraction::integrate(func0);
  const REAL mass_to_test_1 = CompositeInteraction::integrate(func1);

  const REAL error0 = minimum_absrel_error(mass_correct_0, mass_to_test_0);
  const REAL error1 = minimum_absrel_error(mass_correct_1, mass_to_test_1);

  ASSERT_TRUE(error0 < 1.0e-10);
  ASSERT_TRUE(error1 < 1.0e-10);

  composite_intersection->free();
  sycl_target->free();
  mesh->free();
}

namespace {
std::map<int, int> get_map_composite_label_to_bnd_exp_index_orig(
    SpatialDomains::MeshGraphSharedPtr graph,
    std::map<int, std::vector<int>> &boundary_groups,
    MultiRegions::DisContFieldSharedPtr dis_cont_field) {

  // There should be an algorithmically better way to write this function.

  std::map<int, int> map_gid_to_composite_id;
  std::map<int, int> return_map;

  std::set<int> composites_set;
  for (auto vx : boundary_groups) {
    for (int cx : vx.second) {
      composites_set.insert(cx);
    }
  }

  for (int ix : composites_set) {
    return_map[ix] = -1;
  }

  auto graph_composites = graph->GetComposites();
  for (int ix : composites_set) {
    if (graph_composites.count(ix)) {
      auto &geoms = graph_composites.at(ix)->m_geomVec;
      for (auto &geom : geoms) {
        map_gid_to_composite_id[geom->GetGlobalID()] = ix;
      }
    }
  }

  auto bnd_exansions = dis_cont_field->GetBndCondExpansions();

  int index = 0;
  for (auto bx : bnd_exansions) {
    if (bx->GetExpSize()) {
      auto geom_id = bx->GetExp(0)->GetGeom()->GetGlobalID();
      const int composite_id = map_gid_to_composite_id.at(geom_id);

#ifndef NDEBUG
      {
        const int exp_size = bx->GetExpSize();
        for (int ex = 0; ex < exp_size; ex++) {
          const int composite_id_trial = map_gid_to_composite_id.at(
              bx->GetExp(ex)->GetGeom()->GetGlobalID());
          NESOASSERT(
              composite_id_trial == composite_id,
              "Map from boundary index to composite index self check failed.");
        }
      }
#endif

      return_map[composite_id] = index;
    }
    index++;
  }

  return return_map;
}

} // namespace

TEST_P(CompositeInteractionSurfaceFunctionAllD,
       SurfaceFunctionBoundaryConditions) {

  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<int, std::vector<int>> boundary_groups;
  {
    const int maxc = (ndim == 2) ? 4 : 6;
    for (int ix = 1; ix < (maxc + 1); ix++) {
      boundary_groups[ix] = {100 * ix};
    }
  }

  auto prototype_function = std::make_shared<DisContField>(session, graph, "u");
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto composite_intersection = std::make_shared<CompositeIntersection>(
      sycl_target, mesh, boundary_groups, prototype_function);

  auto cibc =
      std::make_shared<CompositeIntersectionBoundaryConditions>(session, graph);

  ASSERT_EQ(graph, cibc->get_mesh_graph());

  auto impl_a = get_map_composite_label_to_bnd_exp_index(graph, boundary_groups,
                                                         prototype_function);

  auto impl_b = get_map_composite_label_to_bnd_exp_index_orig(
      graph, boundary_groups, prototype_function);

  ASSERT_EQ(impl_a, impl_b);

  auto boundary_groups_b = cibc->get_boundary_groups();

  ASSERT_EQ(boundary_groups, boundary_groups_b);

  composite_intersection->free();
  sycl_target->free();
  mesh->free();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleMeshes, CompositeInteractionSurfaceFunctionAllD,
    testing::Values(std::tuple<std::string, std::string, int>(
                        "conditions.xml", "square_triangles_quads.xml", 2),
                    std::tuple<std::string, std::string, double>(
                        "reference_all_types_cube/conditions.xml",
                        "reference_all_types_cube/linear_non_regular_0.5.xml",
                        3)));
