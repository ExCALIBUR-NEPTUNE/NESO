#ifndef _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_HPP_
#define _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_HPP_

#include <string>
#include <vector>

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

namespace NESO::CompositeInteraction {

class CompositeFunction {
protected:
public:
  /// Disable (implicit) copies.
  CompositeFunction(const CompositeFunction &st) = delete;
  /// Disable (implicit) copies.
  CompositeFunction &operator=(CompositeFunction const &a) = delete;
  ~CompositeFunction() = default;

  /// Compute device the function is stored on.
  SYCLTargetSharedPtr sycl_target;
  /// The composite indices this function is defined over.
  std::vector<int> composite_indices;
  /// The Nektar mesh to define functions over
  SpatialDomains::MeshGraphSharedPtr graph;
  /// The function space.
  std::string function_space;
  /// The polynomaial order of the function.
  int num_modes{0};

  /**
   * Create surface function over the specified composites.
   *
   * @param sycl_target Compute device for function.
   * @param composite_indices Elements for function to exist on.
   * @param graph Nektar mesh to define function over.
   * @param function_space Specification of function type, e.g. "DG".
   * @param num_modes Polynomial order of function plus one.
   */
  CompositeFunction(SYCLTargetSharedPtr sycl_target,
                    std::vector<int> composite_indices,
                    SpatialDomains::MeshGraphSharedPtr graph,
                    std::string function_space, int num_modes);
};

} // namespace NESO::CompositeInteraction

#endif
