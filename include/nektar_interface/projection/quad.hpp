#pragma once
#include "basis/basis.hpp"
#include "constants.hpp"
#include "device_data.hpp"
#include "restrict.hpp"
#include "unroll.hpp"


namespace NESO::Project {

//Forward declare
struct ThreadPerCell2D;
struct ThreadPerDof2D;

namespace Private {
struct eQuadBase {
  static constexpr Nektar::LibUtilities::ShapeType shape_type =
      Nektar::LibUtilities::eQuadrilateral;
  static constexpr int dim = 2;
  template <typename T>
  static void loc_coord_to_loc_collapsed(T xi0, T xi1, T &eta0, T &eta1) {
    eta0 = xi0;
    eta1 = xi1;
  };
};
}

template <typename Algorithm>
struct eQuad : public Private::eQuadBase {
};

template <>
struct eQuad<ThreadPerDof2D> : public Private::eQuadBase { 
     
  using algorithm = ThreadPerDof2D;
  template <int nmode, int dim>
  static inline auto NESO_ALWAYS_INLINE local_mem_size() {
    if constexpr (dim == 0 || dim == 1)
      return Constants::gpu_stride * nmode;
    else
      static_assert(true, "second templete parameter must be 0 or 1");
    return -1;
  }


  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1) {
    Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta0, local0);
    Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta1, local1);
    for (int qx = 0; qx < nmode; ++qx) {
      local1[qx * Constants::gpu_stride] *= qoi;
    }
  }

  template <int nmode> static auto NESO_ALWAYS_INLINE get_ndof() {
    return nmode * nmode;
  }

  // TODO: Look at how this would work with vectors
  // As is will not work at all
  template <int nmode, typename T>
  static auto NESO_ALWAYS_INLINE reduce_dof(int idx_local, int count,
                                            T *NESO_RESTRICT mode0,
                                            T *NESO_RESTRICT mode1) {
    int i = idx_local / nmode;
    int j = idx_local % nmode;
    double dof = 0.0;
    for (int k = 0; k < count; ++k) {
      dof += mode1[i * Constants::gpu_stride + k] *
             mode0[j * Constants::gpu_stride + k];
    }
    return dof;
  }
};

template <>
struct eQuad<ThreadPerCell2D> : public Private::eQuadBase { 
  using algorithm = ThreadPerCell2D;
  template <int nmode, typename T, int alpha, int beta>
  static inline NESO_ALWAYS_INLINE void
  project_one_particle(const double eta0, const double eta1, const double qoi,
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
} // namespace NESO::Project
