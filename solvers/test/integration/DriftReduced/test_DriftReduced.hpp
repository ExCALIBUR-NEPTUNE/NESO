#ifndef __NESOSOLVERS_TESTDRIFTREDUCED_HPP__
#define __NESOSOLVERS_TESTDRIFTREDUCED_HPP__

#include <gtest/gtest.h>

#include "EquationSystems/DriftReducedSystem.hpp"
#include "EquationSystems/HW2DSystem.hpp"
#include "solver_test_utils.hpp"
#include "solvers/solver_callback_handler.hpp"
#include "solvers/solver_runner.hpp"

using HWSystem = NESO::Solvers::DriftReduced::HWSystem;

// Growth rate tolerances
constexpr double E_growth_rate_tolerance = 5e-3;
constexpr double W_growth_rate_tolerance = 5e-3;

// Ignore first few steps to allow rate to stabilise
constexpr int first_check_step = 3;

// Mass conservation tolerance
const double mass_cons_tolerance = 2e-12;

/**
 * Struct to calculate and record energy and enstrophy growth rates and compare
 * to expected values
 * (see eqns 18-20 https://rnumata.org/research/materials/turb_ws_jan2006.pdf)
 */
struct CalcHWGrowthRates : public NESO::SolverCallback<HWSystem> {
  std::vector<double> E;
  std::vector<double> W;
  std::vector<double> E_growth_rate_error;
  std::vector<double> W_growth_rate_error;
  std::vector<double> Gamma_a;
  std::vector<double> Gamma_n;
  void call(HWSystem *state) {
    auto md = state->diag_growth_rates_recorder;

    E.push_back(md->compute_energy());
    W.push_back(md->compute_enstrophy());
    Gamma_a.push_back(md->compute_Gamma_a());
    Gamma_n.push_back(md->compute_Gamma_n());

    int nsteps = E.size();
    if (nsteps > first_check_step - 1 && nsteps > 1) {
      int cur_idx = nsteps - 1;
      double dt;
      state->GetSession()->LoadParameter("TimeStep", dt);
      const double avg_Gamma_n =
          0.5 * (Gamma_n[cur_idx - 1] + Gamma_n[cur_idx]);
      const double avg_Gamma_a =
          0.5 * (Gamma_a[cur_idx - 1] + Gamma_a[cur_idx]);
      const double dEdt_exp = avg_Gamma_n - avg_Gamma_a;
      const double dWdt_exp = avg_Gamma_n;
      const double dEdt_act = (E[cur_idx] - E[cur_idx - 1]) / dt;
      const double dWdt_act = (W[cur_idx] - W[cur_idx - 1]) / dt;
      this->E_growth_rate_error.push_back(
          std::abs((dEdt_act - dEdt_exp) / dEdt_exp));
      this->W_growth_rate_error.push_back(
          std::abs((dWdt_act - dWdt_exp) / dWdt_exp));
    }
  }
};

/**
 * Structs to check mass fluid-particle mass conservation
 */
struct CalcMassesPre : public NESO::SolverCallback<HWSystem> {
  void call(HWSystem *state) {
    auto md = state->diag_mass_recorder;
    md->compute_initial_fluid_mass();
  }
};

struct CalcMassesPost : public NESO::SolverCallback<HWSystem> {
  std::vector<double> mass_error;
  void call(HWSystem *state) {
    auto md = state->diag_mass_recorder;
    const double mass_particles = md->compute_particle_mass();
    const double mass_fluid = md->compute_fluid_mass();
    const double mass_total = mass_particles + mass_fluid;
    const double mass_added = md->compute_total_added_mass();
    const double correct_total = mass_added + md->get_initial_mass();
    this->mass_error.push_back(std::fabs(correct_total - mass_total) /
                               std::fabs(correct_total));
  }
};

class HWTest : public NektarSolverTest {
protected:
  void check_growth_rates(bool check_E = true) {
    CalcHWGrowthRates calc_growth_rates_callback;

    MainFuncType runner = [&](int argc, char **argv) {
      SolverRunner solver_runner(argc, argv);
      auto equation_system = std::dynamic_pointer_cast<HWSystem>(
          solver_runner.driver->GetEqu()[0]);

      equation_system->solver_callback_handler.register_post_integrate(
          calc_growth_rates_callback);

      solver_runner.execute();
      solver_runner.finalise();
      return 0;
    };

    int ret_code = run(runner);
    ASSERT_EQ(ret_code, 0);

    if (check_E) {
      ASSERT_THAT(calc_growth_rates_callback.E_growth_rate_error,
                  testing::Each(testing::Le(E_growth_rate_tolerance)));
    }
    ASSERT_THAT(calc_growth_rates_callback.W_growth_rate_error,
                testing::Each(testing::Le(W_growth_rate_tolerance)));
  }

  void check_mass_cons() {
    CalcMassesPre calc_masses_callback_pre;
    CalcMassesPost calc_masses_callback_post;

    MainFuncType runner = [&](int argc, char **argv) {
      SolverRunner solver_runner(argc, argv);
      if (solver_runner.session->DefinesParameter("mass_recording_step")) {
        auto equation_system = std::dynamic_pointer_cast<HWSystem>(
            solver_runner.driver->GetEqu()[0]);

        equation_system->solver_callback_handler.register_pre_integrate(
            calc_masses_callback_pre);
        equation_system->solver_callback_handler.register_post_integrate(
            calc_masses_callback_post);

        solver_runner.execute();
        solver_runner.finalise();
        return 0;
      } else {
        std::cerr << "check_mass_cons callback: session must define "
                     "'mass_recording_step'"
                  << std::endl;
        return 1;
      }
    };

    int ret_code = run(runner);
    ASSERT_EQ(ret_code, 0);
    ASSERT_THAT(calc_masses_callback_post.mass_error,
                testing::Each(testing::Le(mass_cons_tolerance)));
  }

  std::string get_solver_name() override { return "DriftReduced"; }
};

#endif // __NESOSOLVERS_TESTDRIFTREDUCED_HPP__
