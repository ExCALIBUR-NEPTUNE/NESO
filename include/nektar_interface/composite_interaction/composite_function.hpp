#ifndef _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_HPP_
#define _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_HPP_

#include <string>
#include <vector>

#include <MultiRegions/ExpList.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

namespace NESO::CompositeInteraction {

/**
 * TODO
 */
class CompositeFunction {
protected:
  std::shared_ptr<BufferDevice<REAL>> d_dofs;
  std::vector<int> h_dof_offsets;

public:
  /// Disable (implicit) copies.
  CompositeFunction(const CompositeFunction &st) = delete;
  /// Disable (implicit) copies.
  CompositeFunction &operator=(CompositeFunction const &a) = delete;
  ~CompositeFunction() = default;

  /// Compute device the function is stored on.
  SYCLTargetSharedPtr sycl_target;
  /// The expansions that define the function.
  std::vector<MultiRegions::ExpListSharedPtr> exp_lists;

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
                    std::vector<MultiRegions::ExpListSharedPtr> exp_lists);
};

using CompositeFunctionSharedPtr = std::shared_ptr<CompositeFunction>;

} // namespace NESO::CompositeInteraction

#endif
