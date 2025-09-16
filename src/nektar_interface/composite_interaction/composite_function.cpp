#include <nektar_interface/composite_interaction/composite_function.hpp>

namespace NESO::CompositeInteraction {

void CompositeFunction::reset_version() { this->version = 0; }

CompositeFunction::CompositeFunction(
    SYCLTargetSharedPtr sycl_target,
    std::vector<MultiRegions::ExpListSharedPtr> exp_lists, int boundary_group,
    int max_num_dofs)
    : sycl_target(sycl_target), exp_lists(exp_lists),
      boundary_group(boundary_group), max_num_dofs(max_num_dofs) {

  this->h_dof_offsets.resize(exp_lists.size() + 1);

  int num_dofs = 0;

  int index = 0;
  this->total_num_expansions = 0;
  for (auto exp_list : exp_lists) {
    int exp_list_num_dofs = 0;
    if (exp_list) {
      exp_list_num_dofs = exp_list->UpdatePhys().size();
      const int exp_list_size = exp_list->GetExpSize();
      this->total_num_expansions += exp_list_size;
      for (int ex = 0; ex < exp_list_size; ex++) {
        auto exp = exp_list->GetExp(ex);
        NESOASSERT(exp->GetNcoeffs() <= max_num_dofs,
                   "Incompatible DOF stride.");
      }
    }
    this->h_dof_offsets.at(index) = num_dofs;
    num_dofs += exp_list_num_dofs;
    index++;
  }
  this->h_dof_offsets.at(index) = num_dofs;
  this->d_dofs = std::make_shared<BufferDevice<REAL>>(
      this->sycl_target, this->max_num_dofs * this->total_num_expansions);
  this->d_dofs_stage = std::make_shared<BufferDevice<REAL>>(
      this->sycl_target,
      std::max(this->max_num_dofs * this->total_num_expansions, 1));
  this->reset_version();
}

void CompositeFunction::fill(const REAL value) {
  this->sycl_target->queue.fill(this->d_dofs->ptr, value, this->d_dofs->size)
      .wait_and_throw();
}

std::vector<std::vector<std::vector<REAL>>> CompositeFunction::get_dofs() {
  EventStack es;
  const std::size_t num_expansion_lists = this->exp_lists.size();

  std::vector<std::vector<std::vector<REAL>>> h_dofs(num_expansion_lists);

  REAL *d_dofs_ptr = this->d_dofs->ptr;
  for (std::size_t ex = 0; ex < num_expansion_lists; ex++) {
    auto expansion_list = this->exp_lists.at(ex);
    if (expansion_list) {
      const int num_expansions = expansion_list->GetExpSize();
      h_dofs[ex].resize(num_expansions);
      for (int fx = 0; fx < num_expansions; fx++) {
        const int num_dofs_inner = expansion_list->GetExp(fx)->GetNcoeffs();
        h_dofs[ex][fx].resize(num_dofs_inner);
        es.push(sycl_target->queue.memcpy(h_dofs[ex][fx].data(), d_dofs_ptr,
                                          num_dofs_inner * sizeof(REAL)));
        d_dofs_ptr += this->max_num_dofs;
      }
    }
  }

  es.wait();
  return h_dofs;
}

void CompositeFunction::set_dofs(
    std::vector<std::vector<std::vector<REAL>>> &h_dofs) {

  EventStack es;
  const std::size_t num_expansion_lists = this->exp_lists.size();

  REAL *d_dofs_ptr = this->d_dofs->ptr;
  for (std::size_t ex = 0; ex < num_expansion_lists; ex++) {
    auto expansion_list = this->exp_lists.at(ex);
    if (expansion_list) {
      const int num_expansions = expansion_list->GetExpSize();
      for (int fx = 0; fx < num_expansions; fx++) {
        const int num_dofs_inner = expansion_list->GetExp(fx)->GetNcoeffs();
        es.push(sycl_target->queue.memcpy(d_dofs_ptr, h_dofs[ex][fx].data(),
                                          num_dofs_inner * sizeof(REAL)));
        d_dofs_ptr += this->max_num_dofs;
      }
    }
  }

  this->reset_version();
  es.wait();
}

std::vector<REAL> CompositeFunction::get_stage_dofs_linear() {
  return this->d_dofs_stage->get();
}

std::vector<REAL> CompositeFunction::get_dofs_linear() {
  return this->d_dofs->get();
}

} // namespace NESO::CompositeInteraction
