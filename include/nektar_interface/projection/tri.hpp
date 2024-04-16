#pragma once
#include "basis/basis.hpp"
#include "constants.hpp"
#include "device_data.hpp"
#include "restrict.hpp"
#include "unroll.hpp"

namespace NESO::Project {

struct eTriangle {
private:
  // cast from type U to type T
  // special case if U is a sycl vector to call convert function
  // ugly but can't think of anything better
  template <typename T, typename U, typename Q>
  static inline auto NESO_ALWAYS_INLINE convert(U &in) {
    if constexpr (std::is_same<U, cl::sycl::vec<Q, 1>>::value ||
                  std::is_same<U, cl::sycl::vec<Q, 2>>::value ||
                  std::is_same<U, cl::sycl::vec<Q, 4>>::value ||
                  std::is_same<U, cl::sycl::vec<Q, 8>>::value) {
      return in.template convert<T>();
    } else {
      return static_cast<T>(in);
    }
  }
  // solving for
  //(NMODE + 1 + (NMODE +1 - dof))*(dof + 1)/2 = X;
  // if NMODE==4
  // then
  // 0,1,2,3,4 -> 0
  // 5,6,7,8   -> 1
  // 9,10,11   -> 2
  // 12,13     -> 3
  // 14        -> 4
  // i.e. mapping from dof -> index in emodA array
  // TODO: might not be the most efficient way
  // a lookup-table might be better
  // especially in 3D I guess would be cubic in that case?
  template <int NMODE>
  static inline auto NESO_ALWAYS_INLINE calc_tri_row_rev(int dof) {
    double a = double(1 - 2 * (NMODE + 1));
    double n = double(1 + 2 * (dof));
    double tmp = -0.5 * (a + cl::sycl::sqrt(a * a - 4 * n));
    return int(cl::sycl::floor(tmp));
  }

  // Need the abs to work for type sycl::vec
  // but also for bool
  template <typename T> static inline auto NESO_ALWAYS_INLINE to_mask_vec(T a) {
    return cl::sycl::abs(a);
  }

  template <> inline auto NESO_ALWAYS_INLINE to_mask_vec<bool>(bool a) {
    return static_cast<long>(a);
  }

public:
  static constexpr Nektar::LibUtilities::ShapeType shape_type =
      Nektar::LibUtilities::eTriangle;
  static constexpr int dim = 2;
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

  template <typename T>
  inline static void NESO_ALWAYS_INLINE
  loc_coord_to_loc_collapsed(T const xi0, T const xi1, T &eta0, T &eta1) {
    auto d1_origional = T(1.0) - xi1;

    auto zeroTol = T(Constants::Tolerance);
    auto mask_small = to_mask_vec(cl::sycl::fabs(d1_origional) < zeroTol);
    zeroTol = cl::sycl::copysign(zeroTol, d1_origional);
    auto fmask = convert<T, decltype(mask_small), long>(mask_small);
    auto d1 = (T(1.0) - fmask) * d1_origional + fmask * zeroTol;
    eta0 = T(2.0) * (T(1.0) + xi0) / d1 - T(1.0);
    eta1 = xi1;
  }

  template <int nmode, typename T, int alpha, int beta>
  static void NESO_ALWAYS_INLINE fill_local_mem(T eta0, T eta1, T qoi,
                                                T *NESO_RESTRICT local0,
                                                T *NESO_RESTRICT local1) {
    Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta0, local0);
    Basis::eModB<T, nmode, Constants::gpu_stride, alpha, beta>(eta1, local1);
    for (int qx = 0; qx < nmode * (nmode - 1) * 2; ++qx) {
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
    int i = calc_tri_row_rev<nmode>(idx_local);
    double dof = 0.0;
    for (int k = 0; k < count; ++k) {
      // TODO: this correction might be bad or fine (for perf)
      // need to check this
      //(***)
      double correction =
          (idx_local == 1) ? 1.0 : mode0[i * Constants::gpu_stride + k];
      dof += correction * mode1[idx_local * Constants::gpu_stride + k];
    }
    return dof;
  };

  template <int nmode, typename T, int alpha, int beta>
  inline static void NESO_ALWAYS_INLINE project_tpp(const double eta0,
                                                    const double eta1,
                                                    const double qoi,
                                                    double *dofs) {
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
