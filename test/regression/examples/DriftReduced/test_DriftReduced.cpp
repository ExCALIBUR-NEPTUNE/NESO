#include <gtest/gtest.h>

#include "../SolverRegTest.hpp"

namespace NESO::Solvers {

class DriftReducedRegTest : public SolverRegTest {};

/**
 * Regression tests for the DriftReduced solver examples. Note that the test
 * name itself is used to determine the location of the corresponding example
 * directory, config file and mesh file.
 */
TEST_F(DriftReducedRegTest, 2DRogersRicci) { run_and_regress(); }

} // namespace NESO::Solvers