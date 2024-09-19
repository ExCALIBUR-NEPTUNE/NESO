#pragma once
#include "algorithm_types.hpp"
#include "basis/basis.hpp"
#include "constants.hpp"
// #include "device_data.hpp"
#include "restrict.hpp"
#include "unroll.hpp"
#include "util.hpp"
namespace NESO::Project {

namespace Private {
struct eTriangleBase {
private:
public:
  static constexpr Nektar::LibUtilities::ShapeType shape_type =
      Nektar::LibUtilities::eTriangle;
  static constexpr int dim = 2;
  template <typename T>
  inline static void NESO_ALWAYS_INLINE
  loc_coord_to_loc_collapsed(T const xi0, T const xi1, T &eta0, T &eta1) {
    eta0 = Util::Private::collapse_coords(xi0, xi1);
    eta1 = xi1;
  }
};
} // namespace Private

template <typename Algorithm>
struct eTriangle : public Private::eTriangleBase {};

template <> struct eTriangle<ThreadPerDof2D> : public Private::eTriangleBase {
  using algorithm = ThreadPerDof2D;

private:
  // solving for
  //(nmode + 1 + (nmode +1 - dof))*(dof + 1)/2 = X;
  // if nmode==4
  // then
  // 0,1,2,3,4 -> 0
  // 5,6,7,8   -> 1
  // 9,10,11   -> 2
  // 12,13     -> 3
  // 14        -> 4
  // i.e. mapping from dof -> index in emodA array
  template <int nmode>
  static inline auto NESO_ALWAYS_INLINE get_i_from_dof(int dof) {
    double a = double(1 - 2 * (nmode + 1));
    double n = double(1 + 2 * (dof));
    double tmp = -0.5 * (a + cl::sycl::sqrt(a * a - 4 * n));
    return int(cl::sycl::floor(tmp));
  }

public:
  template <int nmode, int dim>
  static inline auto NESO_ALWAYS_INLINE local_mem_size() {
    if constexpr (dim == 0)
      return Constants::gpu_stride * nmode;
    else if constexpr (dim == 1)
      return Constants::gpu_stride * ((nmode * (nmode + 1)) / 2);
    else
      static_assert(true, "dim templete parameter must be 0 or 1");
    return -1;
  }

  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1) {
    Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta0, local0);
    Basis::eModB<T, nmode, Constants::gpu_stride, alpha, beta>(eta1, local1);
    // The correction means the simplest thing is to multiply qoi by the
    // longer array - the alternative is to store the qoi too(!?)
    for (int qx = 0; qx < nmode * (nmode + 1) / 2; ++qx) {
      local1[qx * Constants::gpu_stride] *= qoi;
    }
  }

  template <int nmode> static auto NESO_ALWAYS_INLINE get_ndof() {
    return nmode * (nmode + 1) / 2;
  }

  template <int nmode, typename T>
  static auto NESO_ALWAYS_INLINE reduce_dof(int idx_local, int count,
                                            T *NESO_RESTRICT mode0,
                                            T *NESO_RESTRICT mode1) {
    int i = get_i_from_dof<nmode>(idx_local);
    double dof = 0.0;
    for (int d = 0; d < count; ++d) {
      // TODO: this correction might be bad or fine (for perf)
      // need to check this
      //(***)
      T correction =
          (idx_local == 1) ? T(1.0) : mode0[i * Constants::gpu_stride + d];
      dof += correction * mode1[idx_local * Constants::gpu_stride + d];
    }
    return dof;
  }
};

template <> struct eTriangle<ThreadPerCell2D> : public Private::eTriangleBase {
  using algorithm = ThreadPerCell2D;
  template <int nmode, typename T, int alpha, int beta>
  inline static void NESO_ALWAYS_INLINE
  project_one_particle(T const eta0, T const eta1, T const qoi, T *dofs) {
    T local0[nmode];
    T local1[(nmode * (nmode + 1)) / 2];
    Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta0, local0);
    Basis::eModB<T, nmode, Constants::cpu_stride, alpha, beta>(eta1, local1);
    int mode = 0;
    NESO_UNROLL_LOOP
    for (int i = 0; i < nmode; ++i) {
      NESO_UNROLL_LOOP
      for (int j = 0; j < nmode - i; ++j) {
        double temp = (mode == 1) ? 1.0 : local0[i];
        dofs[mode] += temp * local1[mode] * qoi;
        mode++;
      }
    }
  }
};
} // namespace NESO::Project
