#include <nektar_interface/composite_interaction/composite_function_utility.hpp>

namespace NESO::CompositeInteraction {

REAL integrate(CompositeFunctionSharedPtr func) {

  auto phsvals = func->get_physvals_nektar();

  double value = 0.0;
  for (int exp_listx = 0; exp_listx < func->exp_lists.size(); exp_listx++) {
    auto exp_list = func->exp_lists.at(exp_listx);
    if (exp_list) {
      const int num_expansions = exp_list->GetExpSize();
      for (int ex = 0; ex < num_expansions; ex++) {
        value += exp_list->GetExp(ex)->Integral((*phsvals.at(exp_listx)) +
                                                exp_list->GetPhys_Offset(ex));
      }
    }
  }

  double reduced_value = 0.0;
  MPICHK(MPI_Allreduce(&value, &reduced_value, 1, MPI_DOUBLE, MPI_SUM,
                       func->sycl_target->comm_pair.comm_parent));

  return reduced_value;
}

} // namespace NESO::CompositeInteraction
