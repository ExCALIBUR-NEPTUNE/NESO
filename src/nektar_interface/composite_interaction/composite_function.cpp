#include <nektar_interface/composite_interaction/composite_function.hpp>

namespace NESO::CompositeInteraction {

CompositeFunction::CompositeFunction(
    SYCLTargetSharedPtr sycl_target,
    std::vector<MultiRegions::ExpListSharedPtr> exp_lists)
    : sycl_target(sycl_target), exp_lists(exp_lists) {

  for (auto exp_list : exp_lists) {
    if (exp_list != nullptr) {
      const int exp_list_size = exp_list->GetExpSize();
      for (int ex = 0; ex < exp_list_size; ex++) {
        auto exp = exp_list->GetExp(ex);
      }
    }
  }
}

} // namespace NESO::CompositeInteraction
