#include <gtest/gtest.h>

#include "../SolverRegTest.hpp"

namespace NESO::Solvers {

class DriftPlaneRegTest : public SolverRegTest {};

/**
 * Regression tests for the DriftPlane solver examples. Note that the test name
 * itself is used to determine the location of the corresponding example
 * directory, config file and mesh file.
 */
TEST_F(DriftPlaneRegTest, blob2d) { run_and_regress(); }

} // namespace NESO::Solvers