#include <gtest/gtest.h>

#include "../SolverRegTest.hpp"

namespace NESO::Solvers {

/**
 * @brief Regression tests for the DriftReduced solver examples. Note that test
 * names are used to determine the location of the corresponding example
 * directories and hence the config and mesh files.
 */
class DriftReducedRegTest : public SolverRegTest {};

TEST_F(DriftReducedRegTest, 2DHW) { run_and_regress(); }

/**
Stochasticity from particles isn't accounted for yet, disabling all
with-particles tests for now
TEST_F(DriftReducedRegTest, 2Din3DHW) {
  run_and_regress();
}
 */

TEST_F(DriftReducedRegTest, 2Din3DHW_fluid_only) {
  set_tolerance(1e-7);
  run_and_regress();
}

TEST_F(DriftReducedRegTest, 2DRogersRicci) { run_and_regress(); }

TEST_F(DriftReducedRegTest, 3DHW) {
  set_tolerance(1e-9);
  run_and_regress();
}

} // namespace NESO::Solvers