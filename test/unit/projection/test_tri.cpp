#include <StdRegions/StdExpansion.h>
#include <StdRegions/StdTriExp.h>
#include <StdRegions/StdMatrixKey.h>

#include <gtest/gtest.h>
#include <nektar_interface/projection/algorithm_types.hpp>
#include <nektar_interface/projection/device_data.hpp>

#include <CL/sycl.hpp>
#include <nektar_interface/projection/tri.hpp>
#include <nektar_interface/projection/auto_switch.hpp>

#include "create_data.hpp"

using namespace NESO::Project;
using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::StdRegions;

struct TestData {
    int ndof;
    double val;
    double x,y;
    TestData(int ndof_,double val_, double x_, double y_)
        : ndof(ndof_), val(val_), x(x_), y(y_) {} 
};

class ProjectTriCell : public ::testing::TestWithParam<TestData> { 
    public:
  double
  Integrate(TestData &test_data) {
      cl::sycl::queue Q{cl::sycl::default_selector_v};
      auto const ndof = test_data.ndof;
      int const nmode = ndof - 1;
      PointsKey pk{ndof, eGaussLobattoLegendre};
      BasisKey bk0{eModified_A, nmode, pk};
      BasisKey bk1{eModified_B, nmode, pk};
      StdExpansion *Shape = new StdTriExp{bk0, bk1};
      
      auto data = create_data(Q, nmode*nmode, 
              test_data.val, test_data.x,test_data.y);      
      cl::sycl::event event;
      AUTO_SWITCH( 
              nmode,
              event,
              ThreadPerCell2D::template project,
              FUNCTION_ARGS(data,0,Q),
              double,1,1,NESO::Project::eTriangle<ThreadPerCell2D>);
      event.wait(); 
	  auto buffer = std::vector<double>(Shape->GetNcoeffs(),double(0.0));
	  Q.memcpy(buffer.data(),data.dofs,Shape->GetNcoeffs()*sizeof(double)).wait();
      Array<OneD, double> phi(Shape->GetNcoeffs(), buffer.data());
      Array<OneD, double> coeffs(Shape->GetNcoeffs());
   
      //Multiply by inverse mass matrix
      StdMatrixKey masskey(eInvMass, Shape->DetShapeType(), *Shape);
      DNekMatSharedPtr matsys = Shape->GetStdMatrix(masskey);
      NekVector<NekDouble> coeffsVec(Shape->GetNcoeffs(), coeffs, eWrapper);
      NekVector<NekDouble> phiVec(Shape->GetNcoeffs(), phi, eWrapper); 
      coeffsVec = (*matsys) * phiVec;

      free_data(Q, data);
      //Transfrom to physical space
      Array<OneD, double> phys(Shape->GetTotPoints());
      Shape->BwdTrans(coeffs, phys);
      return Shape->Integral(phys);
  }
};

TEST_P(ProjectTriCell, IntegralIsRight) {
    auto test_data = GetParam();
    ASSERT_NEAR(test_data.val, Integrate(test_data),1.0e-6);
}

INSTANTIATE_TEST_SUITE_P(
        ProjectIntegralTests,
        ProjectTriCell,
        ::testing::Values(
            TestData(3,0.1,0,0),
            TestData(4,5.0,-1,-1),
            TestData(3,1.0,-0.5,1),
            TestData(6,0.1,1,-1),
            TestData(7,12899,0.5,-0.5),
            TestData(5,5.0,-1,-1),
            TestData(3,1.0,0,0),
            TestData(6,0.0,1,-1),
            TestData(8,19,0.5,-0.5)
            ));

class ProjectTriDof : public ::testing::TestWithParam<TestData> { 
    public:
  double
  Integrate(TestData &test_data) {
      cl::sycl::queue Q{cl::sycl::default_selector_v};
      auto const ndof = test_data.ndof;
      int const nmode = ndof - 1;
      PointsKey pk{ndof, eGaussLobattoLegendre};
      BasisKey bk0{eModified_A, nmode, pk};
      BasisKey bk1{eModified_B, nmode, pk};
      StdExpansion *Shape = new StdTriExp{bk0, bk1};
      
      auto data = create_data(Q, nmode*nmode, 
              test_data.val, test_data.x,test_data.y);      
      cl::sycl::event event;
      AUTO_SWITCH( 
              nmode,
              event,
              ThreadPerDof2D::template project,
              FUNCTION_ARGS(data,0,Q),
              double,1,1,NESO::Project::eTriangle<ThreadPerDof2D>);
      event.wait(); 
	  auto buffer = std::vector<double>(Shape->GetNcoeffs(),double(0.0));
	  Q.memcpy(buffer.data(),data.dofs,Shape->GetNcoeffs()*sizeof(double)).wait();
      Array<OneD, double> phi(Shape->GetNcoeffs(), buffer.data());
      Array<OneD, double> coeffs(Shape->GetNcoeffs());
   
      //Multiply by inverse mass matrix
      StdMatrixKey masskey(eInvMass, Shape->DetShapeType(), *Shape);
      DNekMatSharedPtr matsys = Shape->GetStdMatrix(masskey);
      NekVector<NekDouble> coeffsVec(Shape->GetNcoeffs(), coeffs, eWrapper);
      NekVector<NekDouble> phiVec(Shape->GetNcoeffs(), phi, eWrapper); 
      coeffsVec = (*matsys) * phiVec;

      free_data(Q, data);
      //Transfrom to physical space
      Array<OneD, double> phys(Shape->GetTotPoints());
      Shape->BwdTrans(coeffs, phys);
      return Shape->Integral(phys);
  }
};

TEST_P(ProjectTriDof, IntegralIsRight) {
    auto test_data = GetParam();
    ASSERT_NEAR(test_data.val, Integrate(test_data),1.0e-6);
}

INSTANTIATE_TEST_SUITE_P(
        ProjectIntegralTests,
        ProjectTriDof,
        ::testing::Values(
            TestData(6,0.1,1,-1),
            TestData(4,5.0,-1,-1),
            TestData(3,1.0,-0.5,1),
            TestData(3,1.0,-0.5,1),
            TestData(3,1.0,-0.5,1),
            TestData(3,0.1,0,0),
            TestData(6,0.1,1,-1),
            TestData(7,12899,0.5,-0.5),
            TestData(5,5.0,-1,-1),
            TestData(3,1.0,0,0),
            TestData(6,0.0,1,-1)
            ));


