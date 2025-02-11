#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_PYRAMID_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_PYRAMID_HPP
#include "basis/basis.hpp"
#include "restrict.hpp"
#include "unroll.hpp"
#include "util.hpp"
#include <neso_constants.hpp>

namespace NESO::Project {

struct ThreadPerCell;
struct ThreadPerDof;

namespace Private {
template <typename T> constexpr T inline NESO_ALWAYS_INLINE max(T i, T j) {
  return (i > j) ? i : j;
}
struct ePyramidBase {
  // static constexpr Nektar::LibUtilities::ShapeType shape_type =
  //     Nektar::LibUtilities::ePyramid;
  static constexpr int dim = 3;
  template <typename T>
  static inline NESO_ALWAYS_INLINE void

  loc_coord_to_loc_collapsed(T const xi0, T const xi1, T const xi2, T &eta0,
                             T &eta1, T &eta2) {

    // TODO: Doing too much work here now unless compiler is helping
    // i.e. The beginning of the two calls do the same thing
    // Look to refactor (see also the equivilent Tet function)

    eta1 = Util::Private::collapse_coords(xi1, xi2);
    eta0 = Util::Private::collapse_coords(xi0, xi2);
    eta2 = xi2;
  }
  template <int nmode> static auto constexpr NESO_ALWAYS_INLINE get_ndof() {
    return Basis::eModPyrC_len<nmode>();
  }
};
} // namespace Private

template <typename Algorithm> struct ePyramid : public Private::ePyramidBase {};

template <> struct ePyramid<ThreadPerDof> : public Private::ePyramidBase {
  using algorithm = ThreadPerDof;

  template <int nmode, int dim>
  static inline auto NESO_ALWAYS_INLINE local_mem_size(int32_t stride) {
    static_assert(dim >= 0 && dim < 3,
                  "second templete parameter must be 0,1 or 2");
    if constexpr (dim != 2)
      return Basis::eModA_len<nmode>() * stride;
    else
      return Basis::eModPyrC_len<nmode>() * stride;
  }

  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T eta2, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1,
                                                T *NESO_RESTRICT local2,
                                                int32_t stride) {
    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, stride);
    Basis::eModA<T, nmode, alpha, beta>(eta1, local1, stride);
    Basis::eModPyrC<T, nmode, alpha, beta>(eta2, local2, stride);
    for (int qx = 0; qx < Basis::eModPyrC_len<nmode>(); ++qx) {
      local2[qx * stride] *= qoi;
    }
  }

  // TODO: Need a benchmark for how to get the index, migth be bset to
  //  load a look-up table for all the ones that need fiddeling
  //  Could pack it save memory but is local_size smaller that the basis
  //  arrays so so should be ok
  // The inverse this time needs a cubic solving and I don't want to do that
  //  option 1. look up table idx_local -> (i,j,k) (index < 256 even for
  //  nmode==10 so can pack it into chars
  //  option 2. construct a smaller table to work like a switch
  //  e.g. { T_n, T_{n-1},...,1} > scan that
  //  and search for index idx_local is less than
  //  option 3. Newton solve if it works well enough in one step might be ok??
  //  (Probably not) option 4: vvvvvv
  struct indexPair {
    int i, j;
  };

  template <int nmode>
  static constexpr auto indexLookUp = [] {
    std::array<indexPair, get_ndof<nmode>()> a = {};
    int mode = 0;
    for (int i = 0; i < nmode; ++i)
      for (int j = 0; j < nmode; ++j)
        for (int k = 0; k < nmode - Private::max(i, j); ++k)
          a[mode++] = indexPair{i, j};
    return a;
  }();

  // TODO: Look at how this would work with vectors
  // As is will not work at all
  template <int nmode, typename T>
  static auto NESO_ALWAYS_INLINE reduce_dof(int idx_local, int count,
                                            T *NESO_RESTRICT mode0,
                                            T *NESO_RESTRICT mode1,
                                            T *NESO_RESTRICT mode2,
                                            int32_t stride) {
    auto pair = indexLookUp<nmode>[idx_local];
    int i = pair.i;
    int j = pair.j;
    int k = idx_local;
    T dof = 0.0;
    for (int d = 0; d < count; ++d) {
      T temp0 = (k == 1) ? 1.0 : mode0[i * stride + d] * mode1[j * stride + d];
      dof += temp0 * mode2[k * stride + d];
    }
    return dof;
  }
};

template <> struct ePyramid<ThreadPerCell> : public Private::ePyramidBase {
  using algorithm = ThreadPerCell;

  template <int nmode, typename T, int alpha, int beta>
  static inline NESO_ALWAYS_INLINE void
  project_one_particle(const T eta0, const T eta1, const T eta2, const T qoi,
                       T *dofs) {
    T local0[Basis::eModA_len<nmode>()];
    T local1[Basis::eModA_len<nmode>()];
    T local2[Basis::eModPyrC_len<nmode>()];

    Basis::eModA<T, nmode, alpha, beta>(eta0, local0, 1);
    Basis::eModA<T, nmode, alpha, beta>(eta1, local1, 1);
    Basis::eModPyrC<T, nmode, alpha, beta>(eta2, local2, 1);
    int mode = 0;
    NESO_UNROLL_LOOP
    for (int i = 0; i < nmode; ++i) {
      T temp0 = local0[i] * qoi;
      NESO_UNROLL_LOOP
      for (int j = 0; j < nmode; ++j) {
        T temp1 = temp0 * local1[j];
        NESO_UNROLL_LOOP
        for (int k = 0; k < nmode - Private::max(i, j); ++k) {
          auto correction = (mode == 1) ? qoi : temp1;
          *dofs++ += correction * local2[mode];
          mode++;
        }
      }
    }
  }
};

} // namespace NESO::Project
#endif
