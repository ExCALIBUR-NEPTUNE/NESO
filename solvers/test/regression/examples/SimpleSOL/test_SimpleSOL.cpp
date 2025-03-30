#include "test_SimpleSOL.hpp"

#include <gtest/gtest.h>

#include "SimpleSOL.hpp"
/**
 * @brief Regression tests for the SimpleSOL solver examples. Note that test
 * names are used to determine the location of the corresponding example
 * directories and hence the config and mesh files.
 */
namespace NESO::Solvers {

TEST_F(SimpleSOLRegTest, 1D) { run_and_regress(); }

TEST_F(SimpleSOLRegTest, 2D) { run_and_regress(); }

/**
Stochasticity from particles isn't accounted for yet, disabling all
with-particles tests for now
TEST_F(SimpleSOLRegTest, 2DWithParticles) {
  run_and_regress();
}
 */

} // namespace NESO::Solvers