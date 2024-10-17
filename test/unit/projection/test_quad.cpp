#include <StdRegions/StdExpansion.h>
#include <StdRegions/StdMatrixKey.h>
#include <StdRegions/StdQuadExp.h>

#include <gtest/gtest.h>
#include <nektar_interface/projection/algorithm_types.hpp>
#include <nektar_interface/projection/device_data.hpp>

#include <nektar_interface/projection/auto_switch.hpp>
#include <nektar_interface/projection/quad.hpp>
#include <sycl/sycl.hpp>

#include "create_data.hpp"

using namespace NESO::Project;
using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::StdRegions;

class ProjectQuadCell : public ::testing::TestWithParam<TestData2D> {
public:
  double Integrate(TestData2D &test_data) {
    sycl::queue Q{sycl::default_selector_v};
    size_t const ndof = test_data.ndof;
    size_t const nmode = ndof - 1;
    PointsKey pk{ndof, eGaussLobattoLegendre};
    BasisKey bk{eModified_A, nmode, pk};
    StdExpansion *Shape = new StdQuadExp{bk, bk};

    auto [data, pntrs] =
        create_data(Q, nmode * nmode, test_data.val, test_data.x, test_data.y);
    std::optional<sycl::event> event;
    AUTO_SWITCH(static_cast<int>(nmode), event, ThreadPerCell::template project,
                FUNCTION_ARGS(data, 0, Q), double, 1, 1, eQuad<ThreadPerCell>);
    if (event) {
      event.value().wait();
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
    return Shape->Integral(phys);
  }
};

TEST_P(ProjectQuadCell, IntegralIsRight) {
  auto test_data = GetParam();
  ASSERT_NEAR(test_data.val, Integrate(test_data), 1.0e-6);
}

INSTANTIATE_TEST_SUITE_P(ProjectIntegralTests, ProjectQuadCell,
                         ::testing::Values(TestData2D(5, 5.0, -1, -1),
                                           TestData2D(3, 1.0, 0, 0),
                                           TestData2D(6, 0.0, 1, -1),
                                           TestData2D(7, 12899, 0.5, -0.5)));

class ProjectQuadDof : public ::testing::TestWithParam<TestData2D> {
public:
  double Integrate(TestData2D &test_data) {
    sycl::queue Q{sycl::default_selector_v};
    size_t const ndof = test_data.ndof;
    size_t const nmode = ndof - 1;
    PointsKey pk{ndof, eGaussLobattoLegendre};
    BasisKey bk{eModified_A, nmode, pk};
    StdExpansion *Shape = new StdQuadExp{bk, bk};

    auto [data, pntrs] =
        create_data(Q, nmode * nmode, test_data.val, test_data.x, test_data.y);
    std::optional<sycl::event> event;
    AUTO_SWITCH(static_cast<int>(nmode), event, ThreadPerDof::template project,
                FUNCTION_ARGS(data, 0, Q), double, 1, 1, eQuad<ThreadPerDof>);
    if (event) {
      event.value().wait();
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
    return Shape->Integral(phys);
  }
};

TEST_P(ProjectQuadDof, IntegralIsRight) {
  auto test_data = GetParam();
  ASSERT_NEAR(test_data.val, Integrate(test_data), 1.0e-6);
}

INSTANTIATE_TEST_SUITE_P(ProjectIntegralTests, ProjectQuadDof,
                         ::testing::Values(TestData2D(5, 5.0, -1, -1),
                                           TestData2D(3, 1.0, 0, 0),
                                           TestData2D(6, 0.0, 1, -1),
                                           TestData2D(7, 12899, 0.5, -0.5)));
