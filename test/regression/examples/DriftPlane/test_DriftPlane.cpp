#include <gtest/gtest.h>

#include "../SolverRegTest.hpp"

namespace NESO::Solvers {

/**
 * @brief Regression tests for the DriftPlane solver examples. Note that test
 * names are used to determine the location of the corresponding example
 * directories and hence the config and mesh files.
 */
class DriftPlaneRegTest : public SolverRegTest {};

TEST_F(DriftPlaneRegTest, blob2d) { run_and_regress(); }

} // namespace NESO::Solvers