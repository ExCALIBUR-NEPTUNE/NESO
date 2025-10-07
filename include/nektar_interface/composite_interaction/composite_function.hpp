#ifndef _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_HPP_
#define _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_HPP_

#include <string>
#include <vector>

#include <MultiRegions/ExpList.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

namespace NESO::CompositeInteraction {

class CompositeFunctionContext;

/**
 * Type to represent functions defined over multiple composites. Intended use is
 * for functions defined on mesh boundaries.
 */
class CompositeFunction {

  friend class CompositeFunctionContext;

protected:
  // These DOFs assume that each element has max_num_dofs DOFs.
  std::shared_ptr<BufferDevice<REAL>> d_dofs;
  // These DOFs assume that each element has max_num_dofs DOFs.
  std::shared_ptr<BufferDevice<REAL>> d_dofs_stage;
  // The offsets between the expansion lists in the compressed format.
  std::vector<int> h_exp_list_offsets;
  // The offsets between the individual expansions in the compressed format.
  std::shared_ptr<BufferDevice<int>> d_exp_offsets;

  std::int64_t version{0};
  void reset_version();

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
  /// The boundary group the function is defined over.
  int boundary_group{0};
  /// Stride between sets of DOFs
  int max_num_dofs{0};
  /// Total number of elements/expansions across all expansion lists.
  int total_num_expansions{0};
  /// The dimension of the coordinate space of the surface function.
  int ndim{0};

  /**
   * Create surface function over the specified composites.
   *
   * @param sycl_target Compute device for function.
   * @param exp_lists Vector of ExpList instances to create function from.
   * @param boundary_group The boundary group the function is defined on.
   * @param max_num_dofs Stride to use between sets of DOFs.
   */
  CompositeFunction(SYCLTargetSharedPtr sycl_target,
                    std::vector<MultiRegions::ExpListSharedPtr> exp_lists,
                    int boundary_group, int max_num_dofs);

  /**
   * Set all DOFs to a value.
   *
   * @param value Value to set all DOFs to.
   */
  void fill(const REAL value);

  /**
   * @returns DOFs on host.
   */
  std::vector<std::vector<std::vector<REAL>>> get_dofs();

  /**
   * Set the DOFs from a host vector. This function must be called collectively
   * on the communicator.
   *
   * @param h_dofs Host std::vector of length local_dof_count.
   */
  void set_dofs(std::vector<std::vector<std::vector<REAL>>> &h_dofs);

  /**
   * @returns The stage DOF vector on the host.
   */
  std::vector<REAL> get_stage_dofs_linear();

  /**
   * @returns The DOF vector on the host.
   */
  std::vector<REAL> get_dofs_linear();

  /**
   * @returns DOFs (Nektar++ coefficients) in the format Nektar++ usually
   * consumes them.
   */
  std::vector<std::shared_ptr<Array<OneD, NekDouble>>> get_dofs_nektar();

  /**
   * Set DOFs (Nektar++ coefficients) in the format Nektar++ usually uses.
   *
   * @param dofs Input DOFs for each expansion.
   */
  void set_dofs_nektar(
      const std::vector<std::shared_ptr<Array<OneD, NekDouble>>> &dofs);

  /**
   * @returns The PhysVals for the expansions that form the surface function.
   */
  std::vector<std::shared_ptr<Array<OneD, const NekDouble>>>
  get_physvals_nektar();
};

using CompositeFunctionSharedPtr = std::shared_ptr<CompositeFunction>;

} // namespace NESO::CompositeInteraction

#endif
