#include <StdRegions/StdExpansion.h>
#include <StdRegions/StdHexExp.h>
#include <StdRegions/StdMatrixKey.h>

#include <gtest/gtest.h>
#include <nektar_interface/projection/algorithm_types.hpp>
#include <nektar_interface/projection/device_data.hpp>

#include <CL/sycl.hpp>
#include <nektar_interface/projection/auto_switch.hpp>
#include <nektar_interface/projection/hex.hpp>

#include "create_data.hpp"

using namespace NESO::Project;
using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::StdRegions;

struct TestData3D {
  int ndof;
  double val;
  double x, y, z;
  TestData3D(int ndof_, double val_, double x_, double y_, double z_ = 0.0)
      : ndof(ndof_), val(val_), x(x_), y(y_), z{z_} {}
};

class ProjectHexCell : public ::testing::TestWithParam<TestData3D> {
public:
  double Integrate(TestData3D &test_data) {
    cl::sycl::queue Q{cl::sycl::default_selector_v};
    auto const ndof = test_data.ndof;
    int const nmode = ndof - 1;
    PointsKey pk{ndof, eGaussLobattoLegendre};
    BasisKey bk{eModified_A, nmode, pk};
    StdExpansion *Shape = new StdHexExp{bk, bk, bk};

    auto [data, pntrs] = create_data(Q, nmode * nmode * nmode, test_data.val,
                                     test_data.x, test_data.y);
    cl::sycl::event event;
    AUTO_SWITCH(nmode, event, ThreadPerCell3D::template project,
                FUNCTION_ARGS(data, 0, Q), double, 1, 1,
                NESO::Project::eHex<ThreadPerCell3D>);
    event.wait();
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

TEST_P(ProjectHexCell, IntegralIsRight) {
  auto test_data = GetParam();
  ASSERT_NEAR(test_data.val, Integrate(test_data), 1.0e-6);
}

INSTANTIATE_TEST_SUITE_P(
    ProjectIntegralTests, ProjectHexCell,
    ::testing::Values(TestData3D(3, 0.1, 0, 0), TestData3D(4, 5.0, -1, -1),
                      TestData3D(3, 1.0, -0.5, 1), // FAIL
                      TestData3D(6, 0.1, 1, -1),
                      TestData3D(7, 12899, 0.5, -0.5), // FAIL
                      TestData3D(5, 5.0, -1, -1), TestData3D(3, 1.0, 0, 0),
                      TestData3D(6, 0.0, 1, -1),
                      TestData3D(8, 19, 0.5, -0.5) // FAIL
                      ));

class ProjectHexDof : public ::testing::TestWithParam<TestData3D> {
public:
  double Integrate(TestData3D &test_data) {
    cl::sycl::queue Q{cl::sycl::default_selector_v};
    auto const ndof = test_data.ndof;
    int const nmode = ndof - 1;
    PointsKey pk{ndof, eGaussLobattoLegendre};
    BasisKey bk{eModified_A, nmode, pk};
    StdExpansion *Shape = new StdHexExp{bk, bk, bk};

    auto [data, pntrs] = create_data(Q, nmode * nmode * nmode, test_data.val,
                                     test_data.x, test_data.y);

    cl::sycl::event event;
    AUTO_SWITCH(nmode, event, ThreadPerDof3D::template project,
                FUNCTION_ARGS(data, 0, Q), double, 1, 1,
                NESO::Project::eHex<ThreadPerDof3D>);
    event.wait();
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

TEST_P(ProjectHexDof, IntegralIsRight) {
  auto test_data = GetParam();
  ASSERT_NEAR(test_data.val, Integrate(test_data), 1.0e-6);
}

INSTANTIATE_TEST_SUITE_P(
    ProjectIntegralTests, ProjectHexDof,
    ::testing::Values(TestData3D(3, 0.1, 0, 0, 0),
                      TestData3D(4, 5.0, -1, -1, 0.3),
                      TestData3D(3, 1.0, -0.5, 1, -0.3), // FAIL
                      TestData3D(6, 0.1, 1, -1),
                      TestData3D(7, 12899, 0.5, -0.5), // FAIL
                      TestData3D(5, 5.0, -1, -1), TestData3D(3, 1.0, 0, 0),
                      TestData3D(6, 0.0, 1, -1)));
