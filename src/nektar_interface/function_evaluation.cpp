#include <nektar_interface/function_evaluation.hpp>

namespace NESO {

template void
FieldEvaluate<MultiRegions::DisContField>::evaluate(Sym<REAL> sym);
template void FieldEvaluate<MultiRegions::ContField>::evaluate(Sym<REAL> sym);

} // namespace NESO
