#include <nektar_interface/composite_interaction/composite_function.hpp>

namespace NESO::CompositeInteraction {

  CompositeFunction::CompositeFunction(
    SYCLTargetSharedPtr sycl_target,
    std::vector<int> composite_indices,
    SpatialDomains::MeshGraphSharedPtr graph,
    std::string function_space,
    int polynomial_order
  ):
  sycl_target(sycl_target),
  composite_indices(composite_indices),
  graph(graph),
  function_space(function_space),
  polynomial_order(polynomial_order)
  {



  }


}
