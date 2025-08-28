#include <nektar_interface/composite_interaction/composite_function.hpp>

namespace NESO::CompositeInteraction {

CompositeFunction::CompositeFunction(
    SYCLTargetSharedPtr sycl_target,
    std::vector<MultiRegions::ExpListSharedPtr> exp_lists)
    : sycl_target(sycl_target), exp_lists(exp_lists) {}

} // namespace NESO::CompositeInteraction
