#include <nektar_interface/composite_interaction/composite_function_context.hpp>
#include <nektar_interface/expansion_looping/expansion_looping.hpp>

#include <nektar_interface/basis_reference.hpp>
#include <nektar_interface/expansion_looping/jacobi_coeff_mod_basis.hpp>

namespace NESO::CompositeInteraction {

std::map<int, int> get_map_composite_label_to_bnd_exp_index(
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

CompositeFunctionContext::CompositeFunctionContext(
    SYCLTargetSharedPtr sycl_target, SpatialDomains::MeshGraphSharedPtr graph,
    MultiRegions::DisContFieldSharedPtr prototype_field,
    std::map<int, std::vector<int>> boundary_groups)
    : map_composite_label_to_bnd_index(get_map_composite_label_to_bnd_exp_index(
          graph, boundary_groups, prototype_field)),
      sycl_target(sycl_target), graph(graph), prototype_field(prototype_field),
      boundary_groups(boundary_groups)

{

  std::map<int, std::array<int, 2>> map_shape_to_num_modes;
  std::map<int, int> map_shape_to_num_total_modes;

  for (auto shapex : {LibUtilities::eSegment, LibUtilities::eTriangle,
                      LibUtilities::eQuadrilateral}) {
    map_shape_to_num_total_modes[shapex] = 0;
    map_shape_to_num_modes[shapex] = {-1, -1};
    this->map_shape_type_to_total_num_modes[shapex] = {0, 0};
  }

  const std::string error_message =
      "Number of modes differs between elements. This implementation is "
      "not suitable for polynomial order varying between elements.";

  auto lamda_get_num_dim = [&](auto shape_type_int) {
    int num_mode_dims = 2;
    if (shape_type_int == LibUtilities::eSegment) {
      num_mode_dims = 1;
    }
    return num_mode_dims;
  };

  auto lambda_check = [&](const int shape_type_int,
                          const std::array<int, 2> num_modes) {
    const int num_mode_dims = lamda_get_num_dim(shape_type_int);
    auto &current_modes = map_shape_to_num_modes.at(shape_type_int);
    for (int mx = 0; mx < num_mode_dims; mx++) {
      if (current_modes[mx] == -1) {
        // If no num_modes seen set num modes
        current_modes[mx] = num_modes[mx];
      } else {
        NESOASSERT(current_modes[mx] == num_modes[mx], error_message);
      }
    }
  };

  int num_dofs = 0;
  for (auto pair : this->boundary_groups) {
    for (int cx : pair.second) {
      MultiRegions::ExpListSharedPtr exp = nullptr;
      const int index = this->map_composite_label_to_bnd_index.at(cx);
      if (index > -1) {
        auto exp_list = this->prototype_field->GetBndCondExpansions()[index];
        const int num_expansions = exp_list->GetExpSize();
        for (int ex = 0; ex < num_expansions; ex++) {
          auto exp = exp_list->GetExp(ex);
          int shape_type_int = exp->GetGeom()->GetShapeType();
          std::array<int, 2> num_modes = {-1, -1};
          int total_num_modes = 0;
          auto basis = exp->GetBase();
          for (int dx = 0; dx < lamda_get_num_dim(shape_type_int); dx++) {
            num_modes[dx] = exp->GetBasisNumModes(dx);
            const int basis_total_nummodes = basis[dx]->GetTotNumModes();
            total_num_modes += basis_total_nummodes;
            this->map_shape_type_to_total_num_modes.at(shape_type_int).at(dx) =
                basis_total_nummodes;
          }
          lambda_check(shape_type_int, num_modes);
          map_shape_to_num_total_modes[shape_type_int] = total_num_modes;
          num_dofs = std::max(num_dofs, exp->GetNcoeffs());
        }
      }
    }
  }

  auto lambda_global_check = [&](const int shape_type_int) {
    {
      const int num_mode_dims = lamda_get_num_dim(shape_type_int);
      for (int mx = 0; mx < num_mode_dims; mx++) {
        int contrib = map_shape_to_num_modes.at(shape_type_int).at(mx);
        int result = -1;
        MPICHK(MPI_Allreduce(&contrib, &result, 1, MPI_INT, MPI_MAX,
                             this->sycl_target->comm_pair.comm_parent));
        if (contrib > -1) {
          NESOASSERT(result == contrib, error_message);
        } else {
          map_shape_to_num_modes[shape_type_int][mx] = result;
        }
      }
      for (int mx = 0; mx < num_mode_dims; mx++) {
        NESOASSERT(map_shape_to_num_modes.at(shape_type_int).at(mx) ==
                       map_shape_to_num_modes.at(shape_type_int).at(0),
                   "Expected a single value of num modes.");
        int local_total_num_modes =
            map_shape_type_to_total_num_modes[shape_type_int][mx];
        int global_total_num_modes = 0;
        MPICHK(MPI_Allreduce(&local_total_num_modes, &global_total_num_modes, 1,
                             MPI_INT, MPI_MAX,
                             this->sycl_target->comm_pair.comm_parent));
        map_shape_type_to_total_num_modes[shape_type_int][mx] =
            global_total_num_modes;
      }
      this->map_shape_type_to_num_modes[shape_type_int] =
          map_shape_to_num_modes.at(shape_type_int).at(0);
    }

    {
      int contrib = map_shape_to_num_total_modes.at(shape_type_int);
      int result = 0;
      MPICHK(MPI_Allreduce(&contrib, &result, 1, MPI_INT, MPI_MAX,
                           this->sycl_target->comm_pair.comm_parent));
      this->map_shape_type_to_sum_total_num_modes[shape_type_int] = result;
    }
  };

  lambda_global_check(LibUtilities::eSegment);
  lambda_global_check(LibUtilities::eTriangle);
  lambda_global_check(LibUtilities::eQuadrilateral);

  MPICHK(MPI_Allreduce(&num_dofs, &this->max_num_dofs, 1, MPI_INT, MPI_MAX,
                       this->sycl_target->comm_pair.comm_parent));

  // determine the maximum Jacobi order and alpha value required to
  // evaluate the basis functions for these expansions
  int max_alpha = 0;
  int max_n = 0;
  int alpha_tmp = 0;
  int n_tmp = 0;
  for (auto shapex : {LibUtilities::eQuadrilateral, LibUtilities::eTriangle,
                      LibUtilities::eSegment}) {
    BasisReference::get_total_num_modes(
        shapex, map_shape_to_num_modes.at(shapex).at(0), &n_tmp, &alpha_tmp);
    max_alpha = std::max(max_alpha, alpha_tmp);
    max_n = std::max(max_n, n_tmp);
  }

  // Create the Jacobi coefficients and copy them to the device.
  JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);
  this->d_coeffs_pnm10 = std::make_shared<BufferDevice<REAL>>(
      this->sycl_target, jacobi_coeff.coeffs_pnm10);
  this->d_coeffs_pnm11 = std::make_shared<BufferDevice<REAL>>(
      this->sycl_target, jacobi_coeff.coeffs_pnm11);
  this->d_coeffs_pnm2 = std::make_shared<BufferDevice<REAL>>(
      this->sycl_target, jacobi_coeff.coeffs_pnm2);
  this->stride_n = jacobi_coeff.stride_n;
}

CompositeFunctionSharedPtr
CompositeFunctionContext::create_function(const int boundary_group) {

  NESOASSERT(this->boundary_groups.count(boundary_group),
             "Unknown boundary group passed.");

  std::vector<MultiRegions::ExpListSharedPtr> exps;
  exps.reserve(this->boundary_groups.at(boundary_group).size());
  for (int cx : this->boundary_groups.at(boundary_group)) {
    MultiRegions::ExpListSharedPtr exp = nullptr;
    const int index = this->map_composite_label_to_bnd_index.at(cx);
    if (index > -1) {
      // This returns 3D epxansions for the boundary on a 3D mesh?
      // this->prototype_field->GetBndElmtExpansion(index, exp, true);
      exp = this->prototype_field->GetBndCondExpansions()[index];
    }
    exps.push_back(exp);
  }

  return std::make_shared<CompositeFunction>(
      this->sycl_target, exps, boundary_group, this->max_num_dofs);
}

void CompositeFunctionContext::function_project_initialise(
    CompositeFunctionSharedPtr func,
    std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface) {

  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const std::size_t tmp_buffer_size = num_accessible_geoms * this->max_num_dofs;

  func->d_dofs_stage->realloc_no_copy(tmp_buffer_size);
  REAL *k_buffer = func->d_dofs_stage->ptr;

  if (tmp_buffer_size > 0) {
    this->sycl_target->queue.fill(k_buffer, (REAL)0.0, tmp_buffer_size)
        .wait_and_throw();
  }
}

void CompositeFunctionContext::function_project_contribute(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CompositeFunctionSharedPtr func,
    std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface) {

  const bool null_sub_group = particle_sub_group == nullptr;

  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  if (!null_sub_group) {
    auto *k_tree_root = d_tree_root;
    REAL *k_buffer = func->d_dofs_stage->ptr;

    ErrorPropagate ep(this->sycl_target);
    auto k_ep = ep.device_ptr();

    auto lambda_dispatch_2d = [&](const INT shape_type_int, const int num_modes,
                                  auto extract_quantity, auto project_type) {
      const int total_num_modes =
          this->map_shape_type_to_sum_total_num_modes.at(shape_type_int);

      const int max_num_modes0 =
          this->map_shape_type_to_total_num_modes.at(shape_type_int).at(0);
      const int max_num_modes1 =
          this->map_shape_type_to_total_num_modes.at(shape_type_int).at(1);

      auto local_space =
          std::make_shared<LocalMemoryBlock<REAL>>(total_num_modes);
      const int k_max_num_dofs = this->max_num_dofs;

      particle_loop(
          particle_sub_group,
          [=](auto LOCAL_SPACE, auto BOUNDARY_METADATA, auto ELEMENT_TYPE,
              auto REF_COORDS, auto Q) {
            if (ELEMENT_TYPE.at_ephemeral(0) == shape_type_int) {
              const REAL xi[3] = {REF_COORDS.at_ephemeral(0),
                                  REF_COORDS.at_ephemeral(1), 0.0};
              const REAL Q = extract_quantity(Q);

              if (k_tree_root != nullptr) {
                const INT *index;
                bool found = false;
                found =
                    k_tree_root->get(BOUNDARY_METADATA.at_ephemeral(1), &index);
#ifndef NDEBUG
                NESO_KERNEL_ASSERT(found, k_ep);
#endif
                if (found) {
                  REAL *dofs = &k_buffer[(*index) * k_max_num_dofs];

                  REAL *local_space_0 = LOCAL_SPACE.data();
                  REAL *local_space_1 = local_space_0 + max_num_modes0;
                }
              }
            }
          },
          Access::write(local_space),
          Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
          Access::read(Sym<INT>("NESO_BOUNDARY_ELEMENT_TYPE")),
          Access::read(Sym<REAL>("NESO_BOUNDARY_REFERENCE_POSITIONS")),
          Access::read(sym))
          ->execute();
    };

    // TODO
  }
}

void CompositeFunctionContext::function_project_finalise(
    CompositeFunctionSharedPtr func,
    std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface) {}

void CompositeFunctionContext::function_project(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CompositeFunctionSharedPtr func,
    std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface) {

  this->function_project_initialise(func, boundary_mesh_interface);
  this->function_project_contribute(particle_sub_group, sym, component,
                                    is_ephemeral, func,
                                    boundary_mesh_interface);
  this->function_project_finalise(func, boundary_mesh_interface);
}

void CompositeFunctionContext::function_evaluate(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CompositeFunctionSharedPtr func,
    std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface) {

  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const std::size_t tmp_buffer_size = num_accessible_geoms * this->max_num_dofs;

  // TODO CACHING

  func->d_dofs_stage->realloc_no_copy(tmp_buffer_size);
  REAL *k_buffer = func->d_dofs_stage->ptr;

  boundary_mesh_interface->reverse_exchange_from_device(
      func->d_dofs->ptr, this->max_num_dofs, func->d_dofs_stage->ptr);

  const bool null_sub_group = particle_sub_group == nullptr;

  if (!null_sub_group) {
    auto *k_tree_root = d_tree_root;
    REAL *k_buffer = func->d_dofs_stage->ptr;

    ErrorPropagate ep(this->sycl_target);
    auto k_ep = ep.device_ptr();

    auto lambda_dispatch_2d = [&](const INT shape_type_int, auto set_quantity,
                                  const auto loop_type_in) {
      const int num_modes =
          this->map_shape_type_to_num_modes.at(shape_type_int);
      const int total_num_modes =
          this->map_shape_type_to_sum_total_num_modes.at(shape_type_int);
      const int max_num_modes0 =
          this->map_shape_type_to_total_num_modes.at(shape_type_int).at(0);
      const int max_num_modes1 =
          this->map_shape_type_to_total_num_modes.at(shape_type_int).at(1);

      auto local_space =
          std::make_shared<LocalMemoryBlock<REAL>>(total_num_modes);

      const int k_max_num_dofs = this->max_num_dofs;
      const REAL *k_coeffs_pnm10 = this->d_coeffs_pnm10->ptr;
      const REAL *k_coeffs_pnm11 = this->d_coeffs_pnm11->ptr;
      const REAL *k_coeffs_pnm2 = this->d_coeffs_pnm2->ptr;
      const auto k_stride_n = this->stride_n;

      particle_loop(
          particle_sub_group,
          [=](auto LOCAL_SPACE, auto BOUNDARY_METADATA, auto ELEMENT_TYPE,
              auto REF_COORDS, auto Q) {
            if (ELEMENT_TYPE.at_ephemeral(0) == shape_type_int) {
              const REAL xi[3] = {REF_COORDS.at_ephemeral(0),
                                  REF_COORDS.at_ephemeral(1), 0.0};

              if (k_tree_root != nullptr) {
                const INT *index;
                bool found = false;
                found =
                    k_tree_root->get(BOUNDARY_METADATA.at_ephemeral(1), &index);
#ifndef NDEBUG
                NESO_KERNEL_ASSERT(found, k_ep);
#endif
                if (found) {
                  REAL *dofs = &k_buffer[(*index) * k_max_num_dofs];
                  REAL *local_space_0 = LOCAL_SPACE.data();
                  REAL *local_space_1 = local_space_0 + max_num_modes0;
                  REAL *local_space_2 = nullptr;

                  REAL eta0, eta1, eta2;

                  auto loop_type = decltype(loop_type_in)();
                  loop_type.loc_coord_to_loc_collapsed(xi[0], xi[1], xi[2],
                                                       &eta0, &eta1, &eta2);

                  loop_type.evaluate_basis_0(num_modes, eta0, k_stride_n,
                                             k_coeffs_pnm10, k_coeffs_pnm11,
                                             k_coeffs_pnm2, local_space_0);
                  loop_type.evaluate_basis_1(num_modes, eta1, k_stride_n,
                                             k_coeffs_pnm10, k_coeffs_pnm11,
                                             k_coeffs_pnm2, local_space_1);
                  REAL evaluation = 0.0;
                  loop_type.loop_evaluate(num_modes, dofs, local_space_0,
                                          local_space_1, local_space_2,
                                          &evaluation);
                  set_quantity(Q, component, evaluation);
                }
              }
            }
          },
          Access::write(local_space),
          Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
          Access::read(Sym<INT>("NESO_BOUNDARY_ELEMENT_TYPE")),
          Access::read(Sym<REAL>("NESO_BOUNDARY_REFERENCE_POSITIONS")),
          Access::write(sym))
          ->execute();
    };

    if (is_ephemeral) {
      lambda_dispatch_2d(
          LibUtilities::eQuadrilateral,
          [](auto &SYM, const int component, const REAL value) {
            SYM.at_ephemeral(component) = value;
          },
          ExpansionLooping::Quadrilateral{});
      lambda_dispatch_2d(
          LibUtilities::eTriangle,
          [](auto &SYM, const int component, const REAL value) {
            SYM.at_ephemeral(component) = value;
          },
          ExpansionLooping::Triangle{});
    } else {
      lambda_dispatch_2d(
          LibUtilities::eQuadrilateral,
          [](auto &SYM, const int component, const REAL value) {
            SYM.at(component) = value;
          },
          ExpansionLooping::Quadrilateral{});
      lambda_dispatch_2d(
          LibUtilities::eTriangle,
          [](auto &SYM, const int component, const REAL value) {
            SYM.at(component) = value;
          },
          ExpansionLooping::Triangle{});
    }
  }
}

std::vector<INT>
CompositeFunctionContext::get_owned_geoms(const int boundary_group) {

  std::vector<INT> tmp_geoms;
  auto boundary_expansions = this->prototype_field->GetBndCondExpansions();
  for (int cx : this->boundary_groups.at(boundary_group)) {
    const int index = this->map_composite_label_to_bnd_index.at(cx);
    if (index > -1) {
      auto &boundary_expansion = boundary_expansions[index];
      const int num_expansions = boundary_expansion->GetExpSize();
      for (int ex = 0; ex < num_expansions; ex++) {
        const int geom_id =
            boundary_expansion->GetExp(ex)->GetGeom()->GetGlobalID();
        tmp_geoms.push_back(geom_id);
      }
    }
  }

  return tmp_geoms;
}

} // namespace NESO::CompositeInteraction
