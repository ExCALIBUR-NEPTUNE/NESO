#include "nektar_interface/expansion_looping/expansion_looping.hpp"
#include "nektar_interface/function_evaluation.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/DisContField.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace Nektar::MultiRegions;
using namespace NESO::Particles;
using namespace NESO::BasisReference;

static inline REAL local_reduce(const std::vector<REAL> &a,
                                const std::vector<REAL> &b) {
  REAL e = 0.0;
  const int N = a.size();
  for (int ix = 0; ix < N; ix++) {
    e += a[ix] * b[ix];
  }
  return e;
}

static inline void local_scale(const REAL value, std::vector<REAL> &a) {
  const int N = a.size();
  for (int ix = 0; ix < N; ix++) {
    a[ix] *= value;
  }
}

static inline REAL *zero_device_buffer_host(BufferDeviceHost<REAL> &b) {
  for (int ix = 0; ix < b.size; ix++) {
    b.h_buffer.ptr[ix] = 0.0;
  }
  b.host_to_device();
  return b.d_buffer.ptr;
}

static inline REAL *copy_to_device_buffer_host(std::vector<REAL> &v,
                                               BufferDeviceHost<REAL> &b) {
  for (int ix = 0; ix < b.size; ix++) {
    b.h_buffer.ptr[ix] = v[ix];
  }
  b.host_to_device();
  return b.d_buffer.ptr;
}

template <size_t NDIM, ShapeType SHAPE_TYPE, typename BASIS_TYPE>
inline void kernel_basis_wrapper(const int P) {

  int max_alpha, max_n, total_num_modes;

  total_num_modes =
      BasisReference::get_total_num_modes(SHAPE_TYPE, P, &max_n, &max_alpha);
  JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);

  std::map<ShapeType, BasisType> map0;
  map0[eTriangle] = eModified_A;
  map0[eQuadrilateral] = eModified_A;
  map0[eHexahedron] = eModified_A;
  map0[ePyramid] = eModified_A;
  map0[ePrism] = eModified_A;
  map0[eTetrahedron] = eModified_A;
  map0[eSegment] = eModified_A;

  std::map<ShapeType, BasisType> map1;
  map1[eTriangle] = eModified_B;
  map1[eQuadrilateral] = eModified_A;
  map1[eHexahedron] = eModified_A;
  map1[ePyramid] = eModified_A;
  map1[ePrism] = eModified_A;
  map1[eTetrahedron] = eModified_B;

  std::map<ShapeType, BasisType> map2;
  map2[eHexahedron] = eModified_A;
  map2[ePyramid] = eModifiedPyr_C;
  map2[ePrism] = eModified_B;
  map2[eTetrahedron] = eModified_C;

  const int total_num_modes_0 =
      BasisReference::get_total_num_modes(map0.at(SHAPE_TYPE), P);
  const int total_num_modes_1 =
      (NDIM > 1) ? BasisReference::get_total_num_modes(map1.at(SHAPE_TYPE), P)
                 : 1;
  const int total_num_modes_2 =
      (NDIM > 2) ? BasisReference::get_total_num_modes(map2.at(SHAPE_TYPE), P)
                 : 1;

  std::vector<REAL> dir0(total_num_modes_0);
  std::vector<REAL> dir1(total_num_modes_1);
  std::vector<REAL> dir2(total_num_modes_2);
  std::vector<double> to_test(total_num_modes);
  std::vector<double> correct(total_num_modes);
  std::vector<double> coeffs(total_num_modes);
  std::fill(coeffs.begin(), coeffs.end(), 1.0);

  const REAL xi0 = -0.235235;
  const REAL xi1 = -0.565235;
  const REAL xi2 = -0.234;

  REAL eta0, eta1, eta2;

  BASIS_TYPE geom{};

  geom.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1, &eta2);

  geom.evaluate_basis_0(P, eta0, jacobi_coeff.stride_n,
                        jacobi_coeff.coeffs_pnm10.data(),
                        jacobi_coeff.coeffs_pnm11.data(),
                        jacobi_coeff.coeffs_pnm2.data(), dir0.data());

  if (NDIM > 1) {
    geom.evaluate_basis_1(P, eta1, jacobi_coeff.stride_n,
                          jacobi_coeff.coeffs_pnm10.data(),
                          jacobi_coeff.coeffs_pnm11.data(),
                          jacobi_coeff.coeffs_pnm2.data(), dir1.data());
  }

  if (NDIM > 2) {
    geom.evaluate_basis_2(P, eta2, jacobi_coeff.stride_n,
                          jacobi_coeff.coeffs_pnm10.data(),
                          jacobi_coeff.coeffs_pnm11.data(),
                          jacobi_coeff.coeffs_pnm2.data(), dir2.data());
  }

  REAL to_test_evaluate;
  geom.loop_evaluate(P, coeffs.data(), dir0.data(), dir1.data(), dir2.data(),
                     &to_test_evaluate);

  eval_modes(SHAPE_TYPE, P, eta0, eta1, eta2, correct);
  const REAL correct_evaluate = local_reduce(correct, coeffs);

  const REAL err = relative_error(correct_evaluate, to_test_evaluate);
  EXPECT_TRUE(err < 1.0e-14);

  const REAL value = 7.12235;
  local_scale(value, correct);

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  geom.loop_project(P, value, dir0.data(), dir1.data(), dir2.data(),
                    to_test.data());

  BufferDeviceHost<REAL> dh_to_test_dofs(sycl_target, total_num_modes);
  REAL *k_dofs = zero_device_buffer_host(dh_to_test_dofs);

  BufferDeviceHost<REAL> dh_dir0(sycl_target, dir0.size());
  BufferDeviceHost<REAL> dh_dir1(sycl_target, dir1.size());
  BufferDeviceHost<REAL> dh_dir2(sycl_target, dir2.size());
  REAL *k_dir0 = copy_to_device_buffer_host(dir0, dh_dir0);
  REAL *k_dir1 = copy_to_device_buffer_host(dir1, dh_dir1);
  REAL *k_dir2 = copy_to_device_buffer_host(dir2, dh_dir2);

  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.single_task<>([=]() {
          BASIS_TYPE k_geom{};
          k_geom.loop_project(P, value, k_dir0, k_dir1, k_dir2, k_dofs);
        });
      })
      .wait_and_throw();

  dh_to_test_dofs.device_to_host();

  for (int modex = 0; modex < total_num_modes; modex++) {
    const REAL err =
        relative_error(correct[modex], dh_to_test_dofs.h_buffer.ptr[modex]);
    EXPECT_TRUE(err < 1.0e-12);
  }
}

TEST(KernelBasis, Triangle) {
  for (int P = 2; P < 11; P++) {
    kernel_basis_wrapper<2, ShapeType::eTriangle, ExpansionLooping::Triangle>(
        P);
  }
}
TEST(KernelBasis, Quadrilateral) {
  for (int P = 2; P < 11; P++) {
    kernel_basis_wrapper<2, ShapeType::eQuadrilateral,
                         ExpansionLooping::Quadrilateral>(P);
  }
}
TEST(KernelBasis, Hexahedron) {
  for (int P = 2; P < 11; P++) {
    kernel_basis_wrapper<3, ShapeType::eHexahedron,
                         ExpansionLooping::Hexahedron>(P);
  }
}
TEST(KernelBasis, Prism) {
  for (int P = 2; P < 11; P++) {
    kernel_basis_wrapper<3, ShapeType::ePrism, ExpansionLooping::Prism>(P);
  }
}
TEST(KernelBasis, Pyramid) {
  for (int P = 2; P < 11; P++) {
    kernel_basis_wrapper<3, ShapeType::ePyramid, ExpansionLooping::Pyramid>(P);
  }
}
TEST(KernelBasis, Tetrahedron) {
  for (int P = 2; P < 11; P++) {
    kernel_basis_wrapper<3, ShapeType::eTetrahedron,
                         ExpansionLooping::Tetrahedron>(P);
  }
}
TEST(KernelBasis, Segment) {
  for (int P = 2; P < 11; P++) {
    kernel_basis_wrapper<1, ShapeType::eSegment, ExpansionLooping::Segment>(P);
  }
}
