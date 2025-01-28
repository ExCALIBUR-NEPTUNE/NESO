#include "test_SimpleSOL.hpp"

#include <gtest/gtest.h>

#include "SimpleSOL.hpp"

namespace NESO::Solvers {
/**
 * Tests for SimpleSOL solver. Note that the test name itself is used to
 * determine the location of the corresponding example directory, config file
 * and mesh file.
 */
TEST_F(SimpleSOLRegTest, 1D) {
  int ret_code = run({run_SimpleSOL});
  EXPECT_EQ(ret_code, 0);
}

} // namespace NESO::Solvers