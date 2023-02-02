#include <gtest/gtest.h>

#include "SimpleSOL.h"
#include "test_SimpleSOL.h"

/**
 * Tests for SimpleSOL solver. Note that the test name itself is used to
 * determine the locations of the config file, mesh and initial conditions in
 * each case.
 */

// (Pointwise) tolerance to use when comparing profiles
const double tolerance = 5e-3;

TEST_F(SimpleSOLTest, 1D) {
  int ret_code = run({NESO::Solvers::run_SimpleSOL});
  EXPECT_EQ(ret_code, 0);

  // Compare rho, u and T profiles to analytic data
  compare_rho_u_T_profs(tolerance);
}

TEST_F(SimpleSOLTest, 2D) {
  int ret_code = run({NESO::Solvers::run_SimpleSOL});
  EXPECT_EQ(ret_code, 0);

  // Compare rho, u and T profiles to analytic data
  compare_rho_u_T_profs(tolerance);
}

TEST_F(SimpleSOLTest, 2Drot45) {
  int ret_code = run({NESO::Solvers::run_SimpleSOL});
  EXPECT_EQ(ret_code, 0);

  // Compare rho, u and T profiles to analytic data
  compare_rho_u_T_profs(tolerance);
}