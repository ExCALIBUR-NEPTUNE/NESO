#ifndef _NESO_TEST_UNIT_PROJECTION_TEST_COMMON_HPP
#define _NESO_TEST_UNIT_PROJECTION_TEST_COMMON_HPP
#include <StdRegions/StdExpansion.h>
#include <StdRegions/StdHexExp.h>
#include <StdRegions/StdMatrixKey.h>
#include <StdRegions/StdPrismExp.h>
#include <StdRegions/StdPyrExp.h>
#include <StdRegions/StdQuadExp.h>
#include <StdRegions/StdTetExp.h>
#include <StdRegions/StdTriExp.h>
#include <limits>
#include <nektar_interface/projection/shapes.hpp>
#include <utilities/static_case.hpp>

#include "test_common.hpp"

namespace NESO::UnitTest {
using namespace Nektar::LibUtilities;
using namespace Nektar::StdRegions;
namespace Private {

template <template <typename> typename Shape, typename Alg> struct GetNekShape {
  // Don't want this one used
private:
  static Nektar::StdRegions::StdExpansion *value(std::size_t nmode) {
    return nullptr;
  }
};

template <typename Alg> struct GetNekShape<NESO::Project::eQuad, Alg> {
  static Nektar::StdRegions::StdExpansion *value(std::size_t nmode) {
    PointsKey pk{nmode + 1, eGaussLobattoLegendre};
    BasisKey bk{eModified_A, nmode, pk};
    return new StdQuadExp{bk, bk};
  }
};

template <typename Alg> struct GetNekShape<NESO::Project::eTriangle, Alg> {
  static Nektar::StdRegions::StdExpansion *value(std::size_t nmode) {
    PointsKey pk{nmode + 1, eGaussLobattoLegendre};
    BasisKey bk0{eModified_A, nmode, pk};
    BasisKey bk1{eModified_B, nmode, pk};
    return new StdTriExp{bk0, bk1};
  }
};

template <typename Alg> struct GetNekShape<NESO::Project::eHex, Alg> {
  static Nektar::StdRegions::StdExpansion *value(std::size_t nmode) {
    PointsKey pk{nmode + 1, eGaussLobattoLegendre};
    BasisKey bk{eModified_A, nmode, pk};
    return new StdHexExp{bk, bk, bk};
  }
};

template <typename Alg> struct GetNekShape<NESO::Project::ePyramid, Alg> {
  static Nektar::StdRegions::StdExpansion *value(std::size_t nmode) {
    PointsKey pk{nmode + 1, eGaussLobattoLegendre};
    BasisKey bk0{eModified_A, nmode, pk};
    BasisKey bk1{eModifiedPyr_C, nmode, pk};
    return new StdPyrExp{bk0, bk0, bk1};
  }
};

template <typename Alg> struct GetNekShape<NESO::Project::ePrism, Alg> {
  static Nektar::StdRegions::StdExpansion *value(std::size_t nmode) {
    PointsKey pk{nmode + 1, eGaussLobattoLegendre};
    BasisKey bk0{eModified_A, nmode, pk};
    BasisKey bk1{eModified_B, nmode, pk};
    return new StdPrismExp{bk0, bk0, bk1};
  }
};

template <typename Alg> struct GetNekShape<NESO::Project::eTet, Alg> {
  static Nektar::StdRegions::StdExpansion *value(std::size_t nmode) {
    PointsKey pk{nmode + 1, eGaussLobattoLegendre};
    BasisKey bk0{eModified_A, nmode, pk};
    BasisKey bk1{eModified_B, nmode, pk};
    BasisKey bk2{eModified_C, nmode, pk};
    return new StdTetExp{bk0, bk1, bk2};
  }
};
} // namespace Private

template <template <typename> typename S, typename Alg>
inline double integrate_impl(TestData &test_data) {

  using namespace Nektar;
  using Sh = S<Alg>;
  sycl::queue Q{sycl::default_selector_v};
  size_t const nmode = test_data.nmode;
  // size_t const nmode = ndof - 1;
  auto Shape = Private::GetNekShape<S, Alg>::value(nmode);
  auto [data_, pntrs] = create_data<Sh::dim>(
      Q, test_data); // test_data.val, test_data.x, test_data.y);
  // OpenMP can't capture data_ for reasons? so need to copy it
  auto data = data_;
  std::optional<sycl::event> event;

  Utilities::static_case<Constants::min_nummodes, Constants::max_nummodes>(
      nmode, [&](auto I) {
        event = Alg::template project<I, double, Constants::alpha,
                                      Constants::beta, Sh>(data, 0, Q);
      });

  if (event) {
    event.value().wait();
  } else {
    return std::numeric_limits<double>::max();
  }
  auto buffer = std::vector<double>(Shape->GetNcoeffs(), double(0.0));
  Q.memcpy(buffer.data(), data.dofs, Shape->GetNcoeffs() * sizeof(double))
      .wait();
  Array<OneD, double> phi(Shape->GetNcoeffs(), buffer.data());
  Array<OneD, double> coeffs(Shape->GetNcoeffs());

  // Multiply by inverse mass matrix
  StdMatrixKey masskey(eInvMass, Shape->DetShapeType(), *Shape);
  DNekMatSharedPtr matsys = Shape->GetStdMatrix(masskey);
  NekVector<NekDouble> coeffsVec(Shape->GetNcoeffs(), coeffs, eWrapper);
  NekVector<NekDouble> phiVec(Shape->GetNcoeffs(), phi, eWrapper);
  coeffsVec = (*matsys) * phiVec;
  free_data(Q, pntrs);
  // Transfrom to physical space
  Array<OneD, double> phys(Shape->GetTotPoints());
  Shape->BwdTrans(coeffs, phys);
  auto ans = Shape->Integral(phys);
  delete Shape;
  return ans;
}
} // namespace NESO::UnitTest
#endif
