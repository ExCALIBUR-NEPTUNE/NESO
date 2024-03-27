#pragma once
#include "../basis/basis.hpp"
#include "constants.hpp"
#include "device_data.hpp"
#include "unroll.hpp"
namespace NESO::Project {

struct eQuad {
  static constexpr Nektar::LibUtilities::ShapeType shape_type =
      Nektar::LibUtilities::eQuadrilateral;
  static constexpr int dim = 2;
  template <typename T>
  static void loc_coord_to_loc_collapsed(T xi0, T xi1, T &eta0, T &eta1){
    eta0 = xi0; 
    eta1 = xi1;
  };

  
  template <int nmode, typename T, int alpha, int beta>
  static inline NESO_ALWAYS_INLINE void project_tpp(const double eta0,
                                            const double eta1, const double qoi,
                                            double *dofs) {
    T local0[nmode];
    T local1[nmode];
    Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta0, local0);
    Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta1, local1);
    NESO_UNROLL_LOOP
    for (int i = 0; i < nmode; ++i) {
      double temp = local1[i] * qoi;
      NESO_UNROLL_LOOP
      for (int j = 0; j < nmode; ++j) {
        dofs[j + nmode * i] += temp * local0[j];
      }
    }
  }
};
} // namespace NESO::Project::Shape
