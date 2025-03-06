#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_HEX_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_HEX_HPP
#include "basis/basis.hpp"
#include "restrict.hpp"
#include "util.hpp"
#include <neso_constants.hpp>
#include <utilities/unroll.hpp>

namespace NESO::Project {

struct ThreadPerCell;
struct ThreadPerDof;

namespace Private {
struct eHexBase {
  // static constexpr Nektar::LibUtilities::ShapeType shape_type =
  //     Nektar::LibUtilities::eHexahedron;
  static constexpr int dim = 3;
  template <typename T>
  static inline NESO_ALWAYS_INLINE void
  loc_coord_to_loc_collapsed(T const xi0, T const xi1, T const xi2, T &eta0,
                             T &eta1, T &eta2) {
    eta0 = xi0;
    eta1 = xi1;
    eta2 = xi2;
  }
  template <int nmode> static constexpr auto NESO_ALWAYS_INLINE get_ndof() {
    return nmode * nmode * nmode;
  }
};
} // namespace Private

template <typename Algorithm> struct eHex : public Private::eHexBase {};

template <> struct eHex<ThreadPerDof> : public Private::eHexBase {
  using algorithm = ThreadPerDof;
  // Doesn't need one but makes the code cleaner elsewhare
  using lut_type = uint16_t;
  static constexpr bool use_lut = false;
  template <int nmode> static inline lut_type *get_lut(sycl::queue &) {
    return nullptr;
  }
  template <int nmode, int dim>
  static inline auto NESO_ALWAYS_INLINE local_mem_size(int32_t stride) {
    static_assert(dim >= 0 && dim < 3,
                  "second templete parameter must be 0,1 or 2");
    return stride * Basis::eModA_len<nmode>();
  }

  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T eta2, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1,
                                                T *NESO_RESTRICT local2,
                                                int32_t stride) {
    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, stride);
    Basis::eModA<T, nmode, alpha, beta>(eta1, local1, stride);
    Basis::eModA<T, nmode, alpha, beta>(eta2, local2, stride);
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
                                            T *NESO_RESTRICT mode2,
                                            int32_t stride) {
    int i = idx_local / (nmode * nmode);
    int j = (idx_local % (nmode * nmode)) / nmode;
    int k = idx_local % nmode;
    T dof = 0.0;
    for (int d = 0; d < count; ++d) {
      dof +=
          mode2[i * stride + d] * mode1[j * stride + d] * mode0[k * stride + d];
    }
    return dof;
  }
};

template <> struct eHex<ThreadPerCell> : public Private::eHexBase {
  using algorithm = ThreadPerCell;

  template <int nmode, typename T, int alpha, int beta>
  static inline NESO_ALWAYS_INLINE void
  project_one_particle(const T eta0, const T eta1, const T eta2, const T qoi,
                       T *dofs) {
    T local0[nmode];
    T local1[nmode];
    T local2[nmode];

    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, 1);
    Basis::eModA<T, nmode, alpha, beta>(eta1, local1, 1);
    Basis::eModA<T, nmode, alpha, beta>(eta2, local2, 1);
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
#endif
