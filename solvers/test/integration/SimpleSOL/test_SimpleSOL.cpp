#include <gtest/gtest.h>

#include "test_SimpleSOL.hpp"

namespace NESO::Solvers::SimpleSOL {
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
  int ret_code = run();
  EXPECT_EQ(ret_code, 0);

  // Compare rho, u and T profiles to analytic data
  compare_rho_u_T_profs(prof_tolerance);
}

TEST_F(SimpleSOLTest, 2Drot45) {
  int ret_code = run();
  EXPECT_EQ(ret_code, 0);

  // Compare rho, u and T profiles to analytic data
  compare_rho_u_T_profs(prof_tolerance);
}

TEST_F(SimpleSOLTest, 2DWithParticles) {

  SOLWithParticlesMassConservationPre callback_pre;
  SOLWithParticlesMassConservationPost callback_post;

  MainFuncType runner = [&](int argc, char **argv) {
    SolverRunner solver_runner(argc, argv);
    auto equation_system = std::dynamic_pointer_cast<SOLWithParticlesSystem>(
        solver_runner.driver->GetEqu()[0]);

    equation_system->solver_callback_handler.register_pre_integrate(
        callback_pre);
    equation_system->solver_callback_handler.register_post_integrate(
        callback_post);

    solver_runner.execute();
    solver_runner.finalise();
    return 0;
  };

  int ret_code = run(runner);
  EXPECT_EQ(ret_code, 0);
  ASSERT_THAT(callback_post.mass_error,
              testing::Each(testing::Le(mass_cons_tolerance)));
}
} // namespace NESO::Solvers::SimpleSOL