#include "test_SimpleSOL.hpp"

#include <gtest/gtest.h>

#include "SimpleSOL.hpp"

namespace NESO::Solvers {
/**
 * Regression tests for the SimpleSOL solver examples. Note that the test name
 * itself is used to determine the location of the corresponding example
 * directory, config file and mesh file.
 */
TEST_F(SimpleSOLRegTest, 1D) { run_and_regress(); }

TEST_F(SimpleSOLRegTest, 2D) { run_and_regress(); }

// 2DWithParticles test disabled for now; gcc/intel differ by several per cent.
/*
TEST_F(SimpleSOLRegTest, 2DWithParticles) {
  set_tolerance(0.05);
  run_and_regress();
}
*/

} // namespace NESO::Solvers