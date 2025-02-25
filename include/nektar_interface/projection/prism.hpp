#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_PRISM_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_PRISM_HPP
#include "basis/basis.hpp"
#include "restrict.hpp"
#include "util.hpp"
#include <neso_constants.hpp>
#include <utilities/unroll.hpp>

namespace NESO::Project {

struct ThreadPerCell;
struct ThreadPerDof;

template <int nmode> constexpr auto look_up_table() {
  std::array<int, nmode * Basis::eModB_len<nmode>()> arr;
  int next = 0;
  int offset = 0;
  for (int i = 0; i < nmode; ++i) {
    for (int j = 0; j < nmode; ++j) {
      for (int k = 0; k < nmode - i; ++k) {
        arr[next++] = i;
      }
    }
  }
  return arr;
}

namespace Private {
struct ePrismBase {
  static constexpr int dim = 3;
  template <typename T>
  static inline NESO_ALWAYS_INLINE void
  loc_coord_to_loc_collapsed(T const xi0, T const xi1, T const xi2, T &eta0,
                             T &eta1, T &eta2) {
    eta1 = xi1;
    eta2 = xi2;
    eta0 = Util::Private::collapse_coords(xi0, xi2);
  }
  template <int nmode> static constexpr auto NESO_ALWAYS_INLINE get_ndof() {
    return nmode * nmode * (nmode + 1) / 2;
  }
};
} // namespace Private

template <typename Algorithm> struct ePrism : public Private::ePrismBase {};

template <> struct ePrism<ThreadPerDof> : public Private::ePrismBase {
  using algorithm = ThreadPerDof;

private:
  struct indexTriple {
    int i;
    int j;
    int k;
  };
  template <int nmode>
  static constexpr auto indexLookUp = [] {
    std::array<indexTriple, get_ndof<nmode>()> a = {};
    int mode = 0;
    for (int i = 0; i < nmode; ++i) {
      for (int j = 0; j < nmode; ++j) {
        for (int k = 0; k < nmode - i; ++k) {
          auto const offset = nmode * (2 * nmode - i + 1) * i / 2;
          a[mode] = indexTriple{i, (mode - offset) / (nmode - i),
                                (mode - offset) % (nmode - i) +
                                    (2 * nmode - i + 1) * i / 2};
          mode++;
        }
      }
    }
    return a;
  }();

public:
  template <int nmode, int dim>
  static inline auto NESO_ALWAYS_INLINE local_mem_size(int32_t stride) {
    static_assert(dim >= 0 && dim < 3,
                  "dim templete parameter must be 0,1, or 2");
    if constexpr (dim != 2)
      return stride * Basis::eModA_len<nmode>();
    else
      return stride * Basis::eModB_len<nmode>();
  }

  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T eta2, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1,
                                                T *NESO_RESTRICT local2,
                                                int32_t stride) {
    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, stride);
    Basis::eModA<T, nmode, alpha, beta>(eta1, local1, stride);
    Basis::eModB<T, nmode, alpha, beta>(eta2, local2, stride);
    for (int qx = 0; qx < Basis::eModA_len<nmode>(); ++qx) {
      local1[qx * stride] *= qoi;
    }
  }

  // TODO: Look at how this would work with vectors
  // As is will not work at all
  template <int nmode, typename T>
  static auto NESO_ALWAYS_INLINE reduce_dof(int idx_local, int count,
                                            T *NESO_RESTRICT mode0,
                                            T *NESO_RESTRICT mode1,
                                            T *NESO_RESTRICT mode2,
                                            int32_t stride) {
    auto triple = indexLookUp<nmode>[idx_local];
    int const i = triple.i;
    int const j = triple.j;
    int const k = triple.k;
    T dof = 0.0;
    for (int d = 0; d < count; ++d) {
      T correction = (i == 0 && k == 1) ? T(1.0) : mode0[i * stride + d];
      dof += mode2[k * stride + d] * mode1[j * stride + d] * correction;
    }
    return dof;
  }
};

template <> struct ePrism<ThreadPerCell> : public Private::ePrismBase {
  using algorithm = ThreadPerCell;

  template <int nmode, typename T, int alpha, int beta>
  static inline NESO_ALWAYS_INLINE void
  project_one_particle(const T eta0, const T eta1, const T eta2, const T qoi,
                       T *dofs) {
    T local0[Basis::eModA_len<nmode>()];
    T local1[Basis::eModA_len<nmode>()];
    T local2[Basis::eModB_len<nmode>()];

    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, 1);
    Basis::eModA<T, nmode, alpha, beta>(eta1, local1, 1);
    Basis::eModB<T, nmode, alpha, beta>(eta2, local2, 1);
    int mode_r = 0;
    NESO_UNROLL_LOOP
    for (int i = 0; i < nmode; ++i) {
      NESO_UNROLL_LOOP
      for (int j = 0; j < nmode; ++j) {
        NESO_UNROLL_LOOP
        for (int k = 0; k < nmode - i; ++k) {
          T correction = (i == 0 && k == 1) ? 1.0 : local0[i];
          *dofs++ += qoi * correction * local1[j] * local2[k + mode_r];
        }
      }
      mode_r += nmode - i;
    }
  }
};

} // namespace NESO::Project
#endif
