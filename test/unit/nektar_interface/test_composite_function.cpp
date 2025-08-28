#include <SpatialDomains/MeshGraphIO.h>

#include "../../unit/nektar_interface/test_helper_utilities.hpp"

using namespace CompositeInteraction;

/**
 * Construct the map from composite labels to boundary expansion indices.
 *
 * @param graph The mesh the field is defined over.
 * @param boundary_groups The boundary groups used for particle-composite
 * interactions.
 * @param dis_cont_field Prototype field to create boundary functions from.
 * @returns Map from composite label to index in boundary expansions in the
 * prototype field.
 */
std::map<int, int> map_composite_label_to_bnd_exp_index(
    SpatialDomains::MeshGraphSharedPtr graph,
    std::map<int, std::vector<int>> &boundary_groups,
    DisContFieldSharedPtr dis_cont_field) {

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
      return_map[composite_id] = index;
    }
    index++;
  }

  return return_map;
}

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

  auto func0 = std::make_shared<CompositeFunction>(
      sycl_target, boundary_groups.at(0), graph, "DG", 1);

  auto dis_cont_field = std::make_shared<DisContField>(session, graph, "u");

  auto cid_to_boundary_index = map_composite_label_to_bnd_exp_index(
      graph, boundary_groups, dis_cont_field);

  for (auto cx : cid_to_boundary_index) {
    nprint(cx.first, cx.second);
  }

  sycl_target->free();
}
