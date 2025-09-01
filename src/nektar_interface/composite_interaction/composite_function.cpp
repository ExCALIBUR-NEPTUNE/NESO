#include <nektar_interface/composite_interaction/composite_function.hpp>

namespace NESO::CompositeInteraction {

CompositeFunction::CompositeFunction(
    SYCLTargetSharedPtr sycl_target,
    std::vector<MultiRegions::ExpListSharedPtr> exp_lists)
    : sycl_target(sycl_target), exp_lists(exp_lists) {

  this->h_dof_offsets.resize(exp_lists.size() + 1);

  int num_dofs = 0;

  int index = 0;
  for (auto exp_list : exp_lists) {
    int exp_list_num_dofs = 0;
    if (exp_list) {
      exp_list_num_dofs = exp_list->UpdatePhys().size();
      const int exp_list_size = exp_list->GetExpSize();
      for (int ex = 0; ex < exp_list_size; ex++) {
        auto exp = exp_list->GetExp(ex);
      }
    }
    this->h_dof_offsets.at(index) = num_dofs;
    num_dofs += exp_list_num_dofs;
    index++;
  }
  this->h_dof_offsets.at(index) = num_dofs;
  this->d_dofs =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, num_dofs);
}

} // namespace NESO::CompositeInteraction
