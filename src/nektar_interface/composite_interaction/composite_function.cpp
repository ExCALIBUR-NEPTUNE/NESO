#include <nektar_interface/composite_interaction/composite_function.hpp>

namespace NESO::CompositeInteraction {

CompositeFunction::CompositeFunction(SYCLTargetSharedPtr sycl_target,
                                     std::vector<int> composite_indices,
                                     SpatialDomains::MeshGraphSharedPtr graph,
                                     std::string function_space, int num_modes)
    : sycl_target(sycl_target), composite_indices(composite_indices),
      graph(graph), function_space(function_space), num_modes(num_modes) {
  NESOASSERT(function_space == "DG", "Only implemented for DG");
  NESOASSERT(num_modes >= 1,
             "Only implemented for non-negative polynomial order.");

  // map from composite indices to CompositeSharedPtr
  auto graph_composites = graph->GetComposites();

  for (auto ix : composite_indices) {
    // check the composite of interest exists in the MeshGraph on this rank
    if (graph_composites.count(ix)) {
      auto geoms = graph_composites.at(ix)->m_geomVec;
      for (auto &geom : geoms) {
        auto shape_type = geom->GetShapeType();
        NESOASSERT(
            ((shape_type == LibUtilities::eTriangle ||
              shape_type ==
                  LibUtilities::eQuadrilateral)), // ||
                                                  //((shape_type ==
                                                  //LibUtilities::eSegment)),
            "unexpected composite shape type");
      }
    }
  }
}

} // namespace NESO::CompositeInteraction
