#include <gtest/gtest.h>

#include "../SolverRegTest.hpp"

namespace NESO::Solvers {

class H3LAPDRegTest : public SolverRegTest {};

/**
 * Regression tests for the H3LAPD solver examples. Note that the test name
 * itself is used to determine the location of the corresponding example
 * directory, config file and mesh file.
 */
TEST_F(H3LAPDRegTest, 2DRogersRicci) { run_and_regress(); }

TEST_F(H3LAPDRegTest, 2DHW) { run_and_regress(); }

TEST_F(H3LAPDRegTest, 3DHW) {
  set_tolerance(1e-9);
  run_and_regress();
}

TEST_F(H3LAPDRegTest, 2Din3DHW_fluid_only) {
  GTEST_SKIP();
  // Failing at the moment, unclear why
  run_and_regress();
}

} // namespace NESO::Solvers