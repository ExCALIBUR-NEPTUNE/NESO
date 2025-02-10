#include <StdRegions/StdExpansion.h>
#include <StdRegions/StdMatrixKey.h>
#include <StdRegions/StdTetExp.h>

#include <gtest/gtest.h>
#include <nektar_interface/projection/algorithm_types.hpp>
#include <nektar_interface/projection/device_data.hpp>

#include <nektar_interface/projection/tet.hpp>
#include <neso_particles/sycl_typedefs.hpp>

#include "create_data.hpp"
#include "test_common.hpp"

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::StdRegions;
using namespace NESO;

class ProjectTetCell : public ::testing::TestWithParam<TestData> {
public:
  double Integrate(TestData &test_data) {
    return UnitTest::integrate_impl<Project::eTet, Project::ThreadPerCell>(
        test_data);
  }
};

TEST_P(ProjectTetCell, IntegralIsRight) {
  auto test_data = GetParam();
  ASSERT_NEAR(test_data.val, Integrate(test_data), test_data.val * UnitTest::test_tol);
}

INSTANTIATE_TEST_SUITE_P(
    ProjectIntegralTests, ProjectTetCell,
    ::testing::Values(TestData(3, 0.1, 0, 0, 0), TestData(4, 5.0, -1, -1, 0.3),
                      TestData(3, 1.0, -0.5, 1, -0.3), TestData(6, 0.1, 1, -1),
                      TestData(7, 12899, 0.5, -0.5), TestData(5, 5.0, -1, -1),
                      TestData(3, 1.0, 0, 0), TestData(6, 0.0, 1, -1)));

class ProjectTetDof : public ::testing::TestWithParam<TestData> {
public:
  double Integrate(TestData &test_data) {
    return UnitTest::integrate_impl<Project::eTet, Project::ThreadPerDof>(
        test_data);
  }
};

TEST_P(ProjectTetDof, IntegralIsRight) {
  auto test_data = GetParam();
  ASSERT_NEAR(test_data.val, Integrate(test_data), test_data.val * UnitTest::test_tol);
}

INSTANTIATE_TEST_SUITE_P(
    ProjectIntegralTests, ProjectTetDof,
    ::testing::Values(TestData(3, 0.1, 0, 0), TestData(4, 5.0, -1, -1),
                      TestData(3, 1.0, -0.5, 1), TestData(6, 0.1, 1, -1),
                      TestData(7, 12899, 0.5, -0.5), TestData(5, 5.0, -1, -1),
                      TestData(3, 1.0, 0, 0), TestData(6, 0.0, 1, -1),
                      TestData(8, 19, 0.5, -0.5)));
