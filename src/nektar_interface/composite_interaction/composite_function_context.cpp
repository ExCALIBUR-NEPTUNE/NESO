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

{}

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
