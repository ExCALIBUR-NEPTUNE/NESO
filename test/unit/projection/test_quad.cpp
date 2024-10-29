#include <StdRegions/StdExpansion.h>
#include <StdRegions/StdMatrixKey.h>
#include <StdRegions/StdQuadExp.h>

#include <gtest/gtest.h>
#include <nektar_interface/projection/algorithm_types.hpp>
#include <nektar_interface/projection/device_data.hpp>

#include <nektar_interface/projection/quad.hpp>
#include <neso_constants.hpp>
#include <sycl_typedefs.hpp>
#include <utilities/static_case.hpp>

#include "create_data.hpp"
#include "test_common.hpp"

using namespace NESO;
using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::StdRegions;

class ProjectQuadCell : public ::testing::TestWithParam<TestData> {
public:
  double Integrate(TestData &test_data) {
    return UnitTest::integrate_impl<Project::eQuad, Project::ThreadPerCell>(
        test_data);
  }
};

TEST_P(ProjectQuadCell, IntegralIsRight) {
  auto test_data = GetParam();
  ASSERT_NEAR(test_data.val, Integrate(test_data), test_data.val * 1.0e-8);
}

INSTANTIATE_TEST_SUITE_P(ProjectIntegralTests, ProjectQuadCell,
                         ::testing::Values(TestData(5, 5.0, -1, -1),
                                           TestData(3, 1.0, 0, 0),
                                           TestData(6, 0.0, 1, -1),
                                           TestData(7, 12899, 0.5, -0.5)));

class ProjectQuadDof : public ::testing::TestWithParam<TestData> {
public:
  double Integrate(TestData &test_data) {
    return UnitTest::integrate_impl<Project::eQuad, Project::ThreadPerDof>(
        test_data);
  }
};

TEST_P(ProjectQuadDof, IntegralIsRight) {
  auto test_data = GetParam();
  ASSERT_NEAR(test_data.val, Integrate(test_data), test_data.val * 1.0e-8);
}

INSTANTIATE_TEST_SUITE_P(ProjectIntegralTests, ProjectQuadDof,
                         ::testing::Values(TestData(5, 5.0, -1, -1),
                                           TestData(3, 1.0, 0, 0),
                                           TestData(6, 0.0, 1, -1),
                                           TestData(7, 12899, 0.5, -0.5)));
