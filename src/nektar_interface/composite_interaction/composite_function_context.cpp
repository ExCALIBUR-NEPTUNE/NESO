#include <nektar_interface/composite_interaction/composite_function_context.hpp>

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
  map_shape_to_num_modes[static_cast<int>(LibUtilities::eSegment)] = {-1, -1};
  map_shape_to_num_modes[static_cast<int>(LibUtilities::eTriangle)] = {-1, -1};
  map_shape_to_num_modes[static_cast<int>(LibUtilities::eQuadrilateral)] = {-1,
                                                                            -1};

  std::map<int, int> map_shape_to_num_total_modes;
  map_shape_to_num_total_modes[static_cast<int>(LibUtilities::eSegment)] = 0;
  map_shape_to_num_total_modes[static_cast<int>(LibUtilities::eTriangle)] = 0;
  map_shape_to_num_total_modes[static_cast<int>(LibUtilities::eQuadrilateral)] =
      0;

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
          }
          map_shape_to_num_total_modes[shape_type_int] = total_num_modes;
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
        }
      }
      for (int mx = 1; mx < num_mode_dims; mx++) {
        NESOASSERT(map_shape_to_num_modes.at(shape_type_int).at(mx) ==
                       map_shape_to_num_modes.at(shape_type_int).at(0),
                   "Expected a single value of num modes.");
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

  return std::make_shared<CompositeFunction>(this->sycl_target, exps);
}

void CompositeFunctionContext::function_project(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CompositeFunctionSharedPtr func,
    std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface) {

  auto lambda_dispatch_2d = [&](const INT shape_type_int, const int num_modes,
                                auto get_quantity_lambda, auto project_type) {
    const int total_num_modes =
        this->map_shape_type_to_sum_total_num_modes.at(shape_type_int);

    auto local_space =
        std::make_shared<LocalMemoryBlock<REAL>>(total_num_modes);

    particle_loop(
        particle_sub_group,
        [=](auto LOCAL_SPACE, auto ELEMENT_TYPE, auto REF_COORDS, auto Q) {
          if (ELEMENT_TYPE.at_ephemeral(0) == shape_type_int) {
            const REAL xi0 = REF_COORDS.at_ephemeral(0);
            const REAL xi1 = REF_COORDS.at_ephemeral(1);
            const REAL Q = get_quantity_lambda(Q);
          }
        },
        Access::write(local_space),
        Access::read(Sym<INT>("NESO_BOUNDARY_ELEMENT_TYPE")),
        Access::read(Sym<REAL>("NESO_BOUNDARY_REFERENCE_POSITIONS")),
        Access::read(sym))
        ->execute();
  };

  // TODO
}

void CompositeFunctionContext::function_evaluate(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CompositeFunctionSharedPtr func,
    std::shared_ptr<BoundaryMeshInterface> boundary_mesh_interface) {

  // TODO
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
