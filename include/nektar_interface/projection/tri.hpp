#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_TRI_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_TRI_HPP
#include "algorithm_types.hpp"
#include "basis/basis.hpp"
#include <neso_constants.hpp>
// #include "device_data.hpp"
#include "restrict.hpp"
#include "unroll.hpp"
#include "util.hpp"
namespace NESO::Project {

namespace Private {
struct eTriangleBase {
private:
public:
  // static constexpr Nektar::LibUtilities::ShapeType shape_type =
  //     Nektar::LibUtilities::eTriangle;
  static constexpr int dim = 2;
  template <typename T>
  inline static void NESO_ALWAYS_INLINE
  loc_coord_to_loc_collapsed(T const xi0, T const xi1, T &eta0, T &eta1) {
    eta0 = Util::Private::collapse_coords(xi0, xi1);
    eta1 = xi1;
  }
  template <int nmode> static constexpr auto NESO_ALWAYS_INLINE get_ndof() {
    return nmode * (nmode + 1) / 2;
  }
};
} // namespace Private

template <typename Algorithm>
struct eTriangle : public Private::eTriangleBase {};

template <> struct eTriangle<ThreadPerDof> : public Private::eTriangleBase {
  using algorithm = ThreadPerDof;

private:
  template <int nmode>
  static constexpr auto indexLookUp = [] {
    std::array<int, get_ndof<nmode>()> a = {};
    int mode = 0;
    for (int i = 0; i < nmode; ++i) {
      for (int j = 0; j < nmode - i; ++j) {
        a[mode++] = i;
      }
    }
    return a;
  }();
public:
  template <int nmode, int dim>
  static inline constexpr auto NESO_ALWAYS_INLINE
  local_mem_size(int32_t stride) {
    static_assert(dim == 0 || dim == 1,
                  "dim templete parameter must be 0 or 1");
    if constexpr (dim == 0)
      return stride * Basis::eModA_len<nmode>();
    else
      return stride * Basis::eModB_len<nmode>();
  }

  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1,
                                                int32_t stride) {
    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, stride);
    Basis::eModB<T, nmode, alpha, beta>(eta1, local1, stride);
    // The correction means the simplest thing is to multiply qoi by the
    // longer array - the alternative is to store the qoi too(!?)
    for (int qx = 0; qx < Basis::eModB_len<nmode>(); ++qx) {
      local1[qx * stride] *= qoi;
    }
  }

  template <int nmode, typename T>
  static auto NESO_ALWAYS_INLINE reduce_dof(int idx_local, int count,
                                            T *NESO_RESTRICT mode0,
                                            T *NESO_RESTRICT mode1,
                                            int32_t stride) {
    int i = indexLookUp<nmode>[idx_local];
    T dof = T{0.0};
    for (int d = 0; d < count; ++d) {
      T correction = (idx_local == 1) ? T(1.0) : mode0[i * stride + d];
      dof += correction * mode1[idx_local * stride + d];
    }
    return dof;
  }
};

template <> struct eTriangle<ThreadPerCell> : public Private::eTriangleBase {
  using algorithm = ThreadPerCell;
  template <int nmode, typename T, int alpha, int beta>
  inline static void NESO_ALWAYS_INLINE
  project_one_particle(T const eta0, T const eta1, T const qoi, T *dofs) {
    T local0[Basis::eModA_len<nmode>()];
    T local1[Basis::eModB_len<nmode>()];
    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, 1);
    Basis::eModB<T, nmode, alpha, beta>(eta1, local1, 1);
    int mode = 0;
    NESO_UNROLL_LOOP
    for (int i = 0; i < nmode; ++i) {
      NESO_UNROLL_LOOP
      for (int j = 0; j < nmode - i; ++j) {
        T temp = (mode == 1) ? T{1.0} : local0[i];
        dofs[mode] += temp * local1[mode] * qoi;
        mode++;
      }
    }
  }
};
} // namespace NESO::Project
#endif
