#pragma once
#include "basis/basis.hpp"
#include "constants.hpp"
#include "device_data.hpp"
#include "restrict.hpp"
#include "unroll.hpp"

namespace NESO::Project {

// Forward declare
struct ThreadPerCell;
struct ThreadPerDof;

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
  template <int nmode> static auto NESO_ALWAYS_INLINE get_ndof() {
    return nmode * nmode;
  }
};
} // namespace Private

template <typename Algorithm> struct eQuad : public Private::eQuadBase {};

template <> struct eQuad<ThreadPerCell> : public Private::eQuadBase {
  using algorithm = ThreadPerCell;
  template <int nmode, typename T, int alpha, int beta>
  static inline NESO_ALWAYS_INLINE void
  project_one_particle(const T eta0, const T eta1, const T qoi, T *dofs) {
    T local0[Basis::eModA_len<nmode>()];
    T local1[Basis::eModA_len<nmode>()];
    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, 1);
    Basis::eModA<T, nmode, alpha, beta>(eta1, local1, 1);
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

template <> struct eQuad<ThreadPerDof> : public Private::eQuadBase {

  using algorithm = ThreadPerDof;
  template <int nmode, int dim>
  static inline auto NESO_ALWAYS_INLINE local_mem_size(int32_t stride) {
    if constexpr (dim == 0 || dim == 1)
      return stride * Basis::eModA_len<nmode>();
    else
      static_assert(true, "second templete parameter must be 0 or 1");
    return -1;
  }

  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1,
                                                int32_t stride) {
    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, stride);
    Basis::eModA<T, nmode, alpha, beta>(eta1, local1, stride);
    for (int qx = 0; qx < nmode; ++qx) {
      local1[qx * stride] *= qoi;
    }
  }

  // TODO: Look at how this would work with vectors
  // As is will not work at all
  template <int nmode, typename T>
  static auto NESO_ALWAYS_INLINE reduce_dof(int idx_local, int count,
                                            T *NESO_RESTRICT mode0,
                                            T *NESO_RESTRICT mode1,
                                            int32_t stride) {
    int i = idx_local / nmode;
    int j = idx_local % nmode;
    T dof = 0.0;
    for (int k = 0; k < count; ++k) {
      dof += mode1[i * stride + k] * mode0[j * stride + k];
    }
    return dof;
  }
};

} // namespace NESO::Project