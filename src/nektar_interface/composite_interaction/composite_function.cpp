#include <nektar_interface/composite_interaction/composite_function.hpp>

namespace NESO::CompositeInteraction {

void CompositeFunction::reset_version() { this->version = 0; }

CompositeFunction::CompositeFunction(
    SYCLTargetSharedPtr sycl_target,
    std::vector<MultiRegions::ExpListSharedPtr> exp_lists, int boundary_group,
    int max_num_dofs)
    : sycl_target(sycl_target), exp_lists(exp_lists),
      boundary_group(boundary_group), max_num_dofs(max_num_dofs) {

  this->h_exp_list_offsets.resize(exp_lists.size() + 1);
  int num_dofs = 0;

  std::vector<int> h_exp_offsets;
  int exp_offset = 0;

  int index = 0;
  this->total_num_expansions = 0;
  int tmp_ndim = 0;
  for (const auto &exp_list : exp_lists) {
    int exp_list_num_dofs = 0;
    if (exp_list) {
      exp_list_num_dofs = exp_list->GetNcoeffs();
      const int exp_list_size = exp_list->GetExpSize();
      this->total_num_expansions += exp_list_size;
      for (int ex = 0; ex < exp_list_size; ex++) {
        auto exp = exp_list->GetExp(ex);
        NESOASSERT(exp->GetNcoeffs() <= max_num_dofs,
                   "Incompatible DOF stride.");
        h_exp_offsets.push_back(exp_offset);
        exp_offset += exp->GetNcoeffs();
        const auto shape_type = exp->GetGeom()->GetShapeType();
        if (shape_type == LibUtilities::eQuadrilateral) {
          NESOASSERT(exp->GetBasisType(0) == LibUtilities::eModified_A,
                     "Expected eModified_A in direction 0.");
          NESOASSERT(exp->GetBasisType(1) == LibUtilities::eModified_A,
                     "Expected eModified_A in direction 1.");
          tmp_ndim = 2;
        } else if (shape_type == LibUtilities::eTriangle) {
          NESOASSERT(exp->GetBasisType(0) == LibUtilities::eModified_A,
                     "Expected eModified_A in direction 0.");
          NESOASSERT(exp->GetBasisType(1) == LibUtilities::eModified_B,
                     "Expected eModified_B in direction 1.");
          tmp_ndim = 2;
        } else if (shape_type == LibUtilities::eSegment) {
          NESOASSERT(exp->GetBasisType(0) == LibUtilities::eModified_A,
                     "Expected eModified_A in direction 0.");
          tmp_ndim = 1;
        } else {
          NESOASSERT(false, "Unknown boundary shape type.");
        }
      }
    }
    this->h_exp_list_offsets.at(index) = num_dofs;
    num_dofs += exp_list_num_dofs;
    index++;
  }

  MPICHK(MPI_Allreduce(&tmp_ndim, &this->ndim, 1, MPI_INT, MPI_MAX,
                       sycl_target->comm_pair.comm_parent));

  this->h_exp_list_offsets.at(index) = num_dofs;
  h_exp_offsets.push_back(exp_offset);
  this->d_exp_offsets =
      std::make_shared<BufferDevice<int>>(this->sycl_target, h_exp_offsets);

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

std::vector<std::shared_ptr<Array<OneD, NekDouble>>>
CompositeFunction::get_dofs_nektar() {

  auto d_tmp_dofs = get_resource<BufferDevice<REAL>,
                                 ResourceStackInterfaceBufferDevice<NekDouble>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);
  d_tmp_dofs->realloc_no_copy(
      this->h_exp_list_offsets.at(this->h_exp_list_offsets.size() - 1));
  auto k_tmp_dofs = d_tmp_dofs->ptr;
  auto k_exp_offsets = this->d_exp_offsets->ptr;
  auto k_src_dofs = this->d_dofs->ptr;
  auto k_max_num_dofs = this->max_num_dofs;

  auto e0 = this->sycl_target->queue.parallel_for(
      sycl::range<2>(this->total_num_expansions, this->max_num_dofs),
      [=](auto idx) {
        const auto expansion_index = idx.get_id(0);
        const auto dof_index = idx.get_id(1);
        const auto num_dofs =
            k_exp_offsets[expansion_index + 1] - k_exp_offsets[expansion_index];
        if (dof_index < num_dofs) {
          // An implicit conversion from REAL to NekDouble happens here.
          k_tmp_dofs[k_exp_offsets[expansion_index] + dof_index] =
              k_src_dofs[expansion_index * k_max_num_dofs + dof_index];
        }
      });

  std::vector<std::shared_ptr<Array<OneD, NekDouble>>> arrays(
      this->exp_lists.size());

  for (int ex = 0; ex < this->exp_lists.size(); ex++) {
    if (this->exp_lists.at(ex)) {
      arrays[ex] = std::make_shared<Array<OneD, NekDouble>>(
          this->exp_lists.at(ex)->GetNcoeffs());
    }
  }
  e0.wait_and_throw();

  EventStack es;
  for (int ex = 0; ex < this->exp_lists.size(); ex++) {
    if (this->exp_lists.at(ex)) {
      const int num_dofs_in_exp =
          this->h_exp_list_offsets.at(ex + 1) - this->h_exp_list_offsets.at(ex);
      es.push(this->sycl_target->queue.memcpy(
          arrays.at(ex)->data(), k_tmp_dofs + this->h_exp_list_offsets.at(ex),
          sizeof(NekDouble) * num_dofs_in_exp));
    }
  }
  es.wait();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<NekDouble>{}, d_tmp_dofs);
  return arrays;
}

void CompositeFunction::set_dofs_nektar(
    const std::vector<std::shared_ptr<Array<OneD, NekDouble>>> &dofs) {

  auto d_tmp_dofs = get_resource<BufferDevice<REAL>,
                                 ResourceStackInterfaceBufferDevice<NekDouble>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);
  d_tmp_dofs->realloc_no_copy(
      this->h_exp_list_offsets.at(this->h_exp_list_offsets.size() - 1));
  auto k_tmp_dofs = d_tmp_dofs->ptr;
  auto k_exp_offsets = this->d_exp_offsets->ptr;
  auto k_src_dofs = this->d_dofs->ptr;
  auto k_max_num_dofs = this->max_num_dofs;

  EventStack es;
  for (int ex = 0; ex < this->exp_lists.size(); ex++) {
    if (this->exp_lists.at(ex)) {
      const int num_dofs_in_exp =
          this->h_exp_list_offsets.at(ex + 1) - this->h_exp_list_offsets.at(ex);
      es.push(this->sycl_target->queue.memcpy(
          k_tmp_dofs + this->h_exp_list_offsets.at(ex), dofs.at(ex)->data(),
          sizeof(NekDouble) * num_dofs_in_exp));
    }
  }
  es.wait();

  this->sycl_target->queue
      .parallel_for(
          sycl::range<2>(this->total_num_expansions, this->max_num_dofs),
          [=](auto idx) {
            const auto expansion_index = idx.get_id(0);
            const auto dof_index = idx.get_id(1);
            const auto num_dofs = k_exp_offsets[expansion_index + 1] -
                                  k_exp_offsets[expansion_index];
            if (dof_index < num_dofs) {
              // An implicit conversion from NekDouble to REAL happens here.
              k_src_dofs[expansion_index * k_max_num_dofs + dof_index] =
                  k_tmp_dofs[k_exp_offsets[expansion_index] + dof_index];
            }
          })
      .wait_and_throw();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<NekDouble>{}, d_tmp_dofs);
  this->reset_version();
}

std::vector<std::shared_ptr<Array<OneD, const NekDouble>>>
CompositeFunction::get_physvals_nektar() {

  auto nektar_coeffs = this->get_dofs_nektar();
  std::vector<std::shared_ptr<Array<OneD, const NekDouble>>> physvals;

  for (int ex = 0; ex < this->exp_lists.size(); ex++) {
    auto exp_list = this->exp_lists[ex];
    if (exp_list) {
      const int num_phys = exp_list->GetNpoints();
      Array<OneD, NekDouble> tmp_phys_vals(num_phys, 0.0);
      exp_list->BwdTrans(*(nektar_coeffs.at(ex)), tmp_phys_vals);
      physvals.push_back(
          std::make_shared<Array<OneD, const NekDouble>>(tmp_phys_vals));
    } else {
      physvals.push_back(std::make_shared<Array<OneD, const NekDouble>>(0));
    }
  }

  return physvals;
}

} // namespace NESO::CompositeInteraction
