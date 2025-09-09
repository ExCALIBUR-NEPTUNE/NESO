#ifndef _NESO_EXPANSION_LOOPING_LOOP_DATA_HPP_
#define _NESO_EXPANSION_LOOPING_LOOP_DATA_HPP_

#include <neso_particles.hpp>
namespace NP = NESO::Particles;

namespace NESO::PrivateBasisEvaluateBaseKernel {

struct LoopData {
  const int *nummodes;
  const int *coeffs_offsets;
  NP::REAL *global_coeffs;
  int stride_n;
  const NP::REAL *coeffs_pnm10;
  const NP::REAL *coeffs_pnm11;
  const NP::REAL *coeffs_pnm2;
  int ndim;
  int max_total_nummodes0;
  int max_total_nummodes1;
  int max_total_nummodes2;
};

} // namespace NESO::PrivateBasisEvaluateBaseKernel
#endif
