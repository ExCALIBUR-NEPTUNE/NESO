#include <gtest/gtest.h>

#include "../SolverRegTest.hpp"

namespace NESO::Solvers {

/**
 * @brief Regression tests for the H3LAPD solver examples. Note that test names
 * are used to determine the location of the corresponding example directories
 * and hence the config and mesh files.
 */
class H3LAPDRegTest : public SolverRegTest {};

TEST_F(H3LAPDRegTest, 2DHW) { run_and_regress(); }

/**
Stochasticity from particles isn't accounted for yet, disabling all
with-particles tests for now
TEST_F(H3LAPDRegTest, 2Din3DHW) {
  run_and_regress();
}
 */

TEST_F(H3LAPDRegTest, 2Din3DHW_fluid_only) {
  set_tolerance(1e-7);
  run_and_regress();
}

TEST_F(H3LAPDRegTest, 2DRogersRicci) { run_and_regress(); }

TEST_F(H3LAPDRegTest, 3DHW) {
  set_tolerance(1e-9);
  run_and_regress();
}

} // namespace NESO::Solvers