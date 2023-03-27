#include <gtest/gtest.h>

#include "SimpleSOL.h"
#include "test_SimpleSOL.h"

/**
 * Tests for SimpleSOL solver. Note that the test name itself is used to
 * determine the locations of the config file, mesh and initial conditions in
 * each case.
 */

// (Pointwise) tolerance to use when comparing rho, u, T profiles
const double prof_tolerance = 5e-3;
// Mass conservation tolerance
const double mass_cons_tolerance = 1e-14;

TEST_F(SimpleSOLTest, 1D) {
  int ret_code = run({NESO::Solvers::run_SimpleSOL});
  EXPECT_EQ(ret_code, 0);

  // Compare rho, u and T profiles to analytic data
  compare_rho_u_T_profs(prof_tolerance);
}

TEST_F(SimpleSOLTest, 2D) {
  GTEST_SKIP() << "Disabled in favour of SimpleSOLTest.2Drot45";
  int ret_code = run({NESO::Solvers::run_SimpleSOL});
  EXPECT_EQ(ret_code, 0);

  // Compare rho, u and T profiles to analytic data
  compare_rho_u_T_profs(prof_tolerance);
}

TEST_F(SimpleSOLTest, 2Drot45) {
  int ret_code = run({NESO::Solvers::run_SimpleSOL});
  EXPECT_EQ(ret_code, 0);

  // Compare rho, u and T profiles to analytic data
  compare_rho_u_T_profs(prof_tolerance);
}

TEST_F(SimpleSOLTest, 2DWithParticles) {
  int ret_code = run({NESO::Solvers::run_SimpleSOL});
  EXPECT_EQ(ret_code, 0);
  check_mass_conservation(mass_cons_tolerance);
}