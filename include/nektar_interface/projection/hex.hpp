#pragma once
#include "basis/basis.hpp"
#include "constants.hpp"
#include "restrict.hpp"
#include "unroll.hpp"

namespace NESO::Project {

struct ThreadPerCell3D;
struct ThreadPerDof3D;

namespace Private {
struct eHexBase {
  static constexpr Nektar::LibUtilities::ShapeType shape_type =
      Nektar::LibUtilities::eHexahedron;
  static constexpr int dim = 3;
  template <typename T>
  static inline NESO_ALWAYS_INLINE void
  loc_coord_to_loc_collapsed(T const xi0, T const xi1, T const xi2, T &eta0,
                             T &eta1, T &eta2) {
    eta0 = xi0;
    eta1 = xi1;
    eta2 = xi2;
  }
  template <int nmode> static auto NESO_ALWAYS_INLINE get_ndof() {
    return nmode * nmode * nmode;
  }
};
} // namespace Private

template <typename Algorithm> struct eHex : public Private::eHexBase {};

template <> struct eHex<ThreadPerDof3D> : public Private::eHexBase {
  using algorithm = ThreadPerDof3D;
  template <int nmode, int dim>
  static inline auto NESO_ALWAYS_INLINE local_mem_size() {
    if constexpr (dim >= 0 && dim < 3)
      return Constants::gpu_stride * nmode;
    else
      static_assert(true, "second templete parameter must be 0,1 or 2");
    return -1;
  }

  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T eta2, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1,
                                                T *NESO_RESTRICT local2) {
    Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta0, local0);
    Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta1, local1);
    Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta2, local2);
    for (int qx = 0; qx < nmode; ++qx) {
      local1[qx * Constants::gpu_stride] *= qoi;
    }
  }

  // TODO: Look at how this would work with vectors
  // As is will not work at all
  template <int nmode, typename T>
  static auto NESO_ALWAYS_INLINE reduce_dof(int idx_local, int count,
                                            T *NESO_RESTRICT mode0,
                                            T *NESO_RESTRICT mode1,
                                            T *NESO_RESTRICT mode2) {
    int i = idx_local / (nmode * nmode);
    int j = (idx_local % (nmode * nmode)) / nmode;
    int k = idx_local % nmode;
    T dof = 0.0;
    for (int d = 0; d < count; ++d) {
      dof += mode2[i * Constants::gpu_stride + d] *
             mode1[j * Constants::gpu_stride + d] *
             mode0[k * Constants::gpu_stride + d];
    }
    return dof;
  }
};

template <> struct eHex<ThreadPerCell3D> : public Private::eHexBase {
  using algorithm = ThreadPerCell3D;

  template <int nmode, typename T, int alpha, int beta>
  static inline NESO_ALWAYS_INLINE void
  project_one_particle(const T eta0, const T eta1, const T eta2, const T qoi,
                       T *dofs) {
    T local0[nmode];
    T local1[nmode];
    T local2[nmode];

    Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta0, local0);
    Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta1, local1);
    Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta2, local2);
    NESO_UNROLL_LOOP
    for (int i = 0; i < nmode; ++i) {
      T temp0 = local2[i] * qoi;
      NESO_UNROLL_LOOP
      for (int j = 0; j < nmode; ++j) {
        T temp1 = temp0 * local1[j];
        NESO_UNROLL_LOOP
        for (int k = 0; k < nmode; ++k) {
          *dofs++ += temp1 * local0[k];
        }
      }
    }
  }
};

} // namespace NESO::Project
