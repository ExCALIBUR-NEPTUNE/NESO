#pragma once
#include "basis/basis.hpp"
#include "constants.hpp"
#include "restrict.hpp"
#include "unroll.hpp"
#include "util.hpp"

namespace NESO::Project {

struct ThreadPerCell3D;
struct ThreadPerDof3D;

namespace Private {
struct eTetBase {
  static constexpr Nektar::LibUtilities::ShapeType shape_type =
      Nektar::LibUtilities::eTetrahedron;
  static constexpr int dim = 3;
  template <typename T>
  static inline NESO_ALWAYS_INLINE void
  loc_coord_to_loc_collapsed(T const xi0, T const xi1, T const xi2, T &eta0,
                             T &eta1, T &eta2) {

    eta1 = Util::Private::collapse_coords(xi1, xi2);
    // Leaky here re-factor into 2 functions
    eta0 = Util::Private::collapse_coords(xi0, xi1 + xi2 + T(1.0));
    // abstraction
    eta2 = xi2;
  }
};
} // namespace Private

template <typename Algorithm> struct eTet : public Private::eTetBase {};

template <> struct eTet<ThreadPerDof3D> : public Private::eTetBase {
  using algorithm = ThreadPerDof3D;

  template <int nmode, int dim>
  static inline auto NESO_ALWAYS_INLINE local_mem_size() {
    if constexpr (dim == 0)
      return Basis::eModA_len<nmode>() * Constants::gpu_stride;
    else if constexpr (dim == 1)
      return Basis::eModB_len<nmode>() * Constants::gpu_stride;
    else if constexpr (dim == 2)
      return Basis::eModC_len<nmode>() * Constants::gpu_stride;
    else {
      static_assert(true, "second templete parameter must be 0,1 or 2");
      return -1;
    }
  }

  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T eta2, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1,
                                                T *NESO_RESTRICT local2) {
    Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta0, local0);
    Basis::eModB<T, nmode, Constants::gpu_stride, alpha, beta>(eta1, local1);
    Basis::eModC<T, nmode, Constants::gpu_stride, alpha, beta>(eta2, local2);
    for (int qx = 0; qx < Basis::eModC_len<nmode>(); ++qx) {
      local2[qx * Constants::gpu_stride] *= qoi;
    }
  }
  template <int nmode> static auto NESO_ALWAYS_INLINE get_ndof() {
    return nmode * (nmode + 1) * (nmode + 2) / 6;
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
  struct index_pair {
    int i, j;
  };
#warning "TEMP HACK TO GET IT WORKING NEED A GOOD WAY TO GET THE RIGHT INDEX"
  template <int nmode> static auto NESO_ALWAYS_INLINE get_index(int idx_local) {
    int mode = 0;
    struct index_pair pair = {-1, -1};
    for (int i = 0; i < nmode; ++i)
      for (int j = 0; j < nmode - i; ++j)
        for (int k = 0; k < nmode - i - j; ++k)
          pair = (idx_local == mode++) ? (index_pair){i, j} : pair;
    assert(pair.i != -1 && pair.j != -1);
    return pair;
  }
  // TODO: Look at how this would work with vectors
  // As is will not work at all
  template <int nmode, typename T>
  static auto NESO_ALWAYS_INLINE reduce_dof(int idx_local, int count,
                                            T *NESO_RESTRICT mode0,
                                            T *NESO_RESTRICT mode1,
                                            T *NESO_RESTRICT mode2) {
    auto pair = get_index<nmode>(idx_local);
    int i = pair.i;
    int j = (i + 1) * (2 * nmode - i) / 2 - nmode + i + pair.j;
    int k = idx_local;
    T dof = 0.0;
    for (int d = 0; d < count; ++d) {
      T temp0 = (i == 0 && j == 1) ? 1.0 : mode0[i * Constants::gpu_stride + d];
      T temp1 = (k == 1) ? 1.0 : temp0 * mode1[j * Constants::gpu_stride + d];
      dof += temp1 * mode2[k * Constants::gpu_stride + d];
    }
    return dof;
  }
};

template <> struct eTet<ThreadPerCell3D> : public Private::eTetBase {
  using algorithm = ThreadPerCell3D;

  template <int nmode, typename T, int alpha, int beta>
  static inline NESO_ALWAYS_INLINE void
  project_one_particle(const T eta0, const T eta1, const T eta2, const T qoi,
                       T *dofs) {
    T local0[Basis::eModA_len<nmode>()];
    T local1[Basis::eModB_len<nmode>()];
    T local2[Basis::eModC_len<nmode>()];

    Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta0, local0);
    Basis::eModB<T, nmode, Constants::cpu_stride, alpha, beta>(eta1, local1);
    Basis::eModC<T, nmode, Constants::cpu_stride, alpha, beta>(eta2, local2);
    int mode = 0;
    int mode_q = 0;
    NESO_UNROLL_LOOP
    for (int i = 0; i < nmode; ++i) {
      T temp0 = local0[i];
      NESO_UNROLL_LOOP
      for (int j = 0; j < nmode - i; ++j) {
        T temp1 = local1[mode_q];
        assert(mode_q == ((i + 1) * (2 * nmode - i) / 2 - nmode + i + j));
        mode_q++;
        NESO_UNROLL_LOOP
        for (int k = 0; k < nmode - i - j; ++k) {
          auto temp2 = local2[mode];
          if (mode == 1)
            *dofs++ += qoi * temp2;
          else if (i == 0 && j == 1)
            *dofs++ += qoi * temp1 * temp2;
          else
            *dofs++ += qoi * temp0 * temp1 * temp2;

          mode++;
        }
      }
    }
  }
};

} // namespace NESO::Project
