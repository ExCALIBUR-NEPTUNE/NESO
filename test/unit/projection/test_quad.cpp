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

struct TestData {
  int ndof;
  double val;
  double x, y;
  TestData(int ndof_, double val_, double x_, double y_)
      : ndof(ndof_), val(val_), x(x_), y(y_) {}
};

class ProjectQuadCell : public ::testing::TestWithParam<TestData> {
public:
  double Integrate(TestData &test_data) {
    cl::sycl::queue Q{cl::sycl::default_selector_v};
    auto const ndof = test_data.ndof;
    int const nmode = ndof - 1;
    PointsKey pk{ndof, eGaussLobattoLegendre};
    BasisKey bk{eModified_A, nmode, pk};
    StdExpansion *Shape = new StdQuadExp{bk, bk};

    auto data =
        create_data(Q, nmode * nmode, test_data.val, test_data.x, test_data.y);
    cl::sycl::event event;
    AUTO_SWITCH(nmode, event, ThreadPerCell2D::template project,
                FUNCTION_ARGS(data, 0, Q), double, 1, 1,
                eQuad<ThreadPerCell2D>);
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

    free_data(Q, data);
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
                         ::testing::Values(TestData(5, 5.0, -1, -1),
                                           TestData(3, 1.0, 0, 0),
                                           TestData(6, 0.0, 1, -1),
                                           TestData(7, 12899, 0.5, -0.5)));

class ProjectQuadDof : public ::testing::TestWithParam<TestData> {
public:
  double Integrate(TestData &test_data) {
    cl::sycl::queue Q{cl::sycl::default_selector_v};
    auto const ndof = test_data.ndof;
    int const nmode = ndof - 1;
    PointsKey pk{ndof, eGaussLobattoLegendre};
    BasisKey bk{eModified_A, nmode, pk};
    StdExpansion *Shape = new StdQuadExp{bk, bk};

    auto data =
        create_data(Q, nmode * nmode, test_data.val, test_data.x, test_data.y);
    cl::sycl::event event;
    AUTO_SWITCH(nmode, event, ThreadPerDof2D::template project,
                FUNCTION_ARGS(data, 0, Q), double, 1, 1, eQuad<ThreadPerDof2D>);
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

    free_data(Q, data);
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
                         ::testing::Values(TestData(5, 5.0, -1, -1),
                                           TestData(3, 1.0, 0, 0),
                                           TestData(6, 0.0, 1, -1),
                                           TestData(7, 12899, 0.5, -0.5)));

#if 0
TEST(Projection, QuadOneParticleCell) {
  cl::sycl::queue Q{cl::sycl::default_selector_v};
  constexpr int ndof = 5;
  constexpr int nmode = 4;
  double number = 10.0;
  PointsKey pk{ndof, eGaussLobattoLegendre};
  BasisKey bk{eModified_A, nmode, pk};
  StdExpansion *Quad = new StdQuadExp{bk, bk};

  Array<OneD, double> x(Quad->GetTotPoints());
  Array<OneD, double> y(Quad->GetTotPoints());
  auto data = create_data(Q, ndof*ndof, number, x[2],y[2]);

  Quad->GetCoords(x,y);
  
  ThreadPerCell2D::template project<ndof, double, 1, 1, eQuad<ThreadPerCell2D>>(
      data, 0, Q)
      .wait();
  Array<OneD, double> in(ndof * ndof, data.dofs);
  Array<OneD, double> out(ndof * ndof);

  Quad->BwdTrans(in, out);
  auto S = Quad->Integral(out);
  free_data(Q, data);
  std::cout << number << ", " << S << std::endl;
  std::cout << Quad->GetTotPoints() << ", " << Quad->GetNcoeffs() << std::endl;
  EXPECT_DOUBLE_EQ(S, number);
}

TEST(Projection, QuadOneParticleDof) {
  cl::sycl::queue Q{cl::sycl::default_selector_v};
  constexpr int ndof = 5;
  constexpr int nmode = 4;
  double number = 10.0;
  

  PointsKey pk{ndof, eGaussLobattoLegendre};
  BasisKey bk{eModified_A, nmode, pk};
  StdExpansion *Shape = new StdQuadExp{bk, bk};
  Array<OneD, double> x(Shape->GetTotPoints());
  Array<OneD, double> y(Shape->GetTotPoints());


  auto data = create_data(Q, nmode*nmode, number, 0, 0);

  ThreadPerDof2D::template project<nmode, double, 1, 1, eQuad<ThreadPerDof2D>>(
      data, 0, Q)
      .wait();
  Array<OneD, double> in(Shape->GetNcoeffs(), data.dofs);
  Array<OneD, double> inM(Shape->GetNcoeffs());
  Array<OneD, double> out(Shape->GetTotPoints());


  StdMatrixKey masskey(eInvMass, Shape->DetShapeType(), *Shape);
  DNekMatSharedPtr matsys = Shape->GetStdMatrix(masskey);
  NekVector<NekDouble> pVec(Shape->GetNcoeffs(), inM, eWrapper);
  NekVector<NekDouble> inVec(Shape->GetNcoeffs(), in, eWrapper); 
  pVec = (*matsys) * inVec;
  
  Shape->BwdTrans(inM, out);
  auto S = Shape->Integral(out);
  free_data(Q, data);
  std::cout << x[0] << ", " << y[0] << std::endl;
  EXPECT_NEAR(S, number,1.0e-6);
}
#endif
