#ifndef H3LAPD_TESTS_H
#define H3LAPD_TESTS_H

#include <gtest/gtest.h>

#include "EquationSystems/DriftReducedSystem.hpp"
#include "EquationSystems/HW2Din3DSystem.hpp"
#include "H3LAPD.hpp"
#include "solver_test_utils.h"
#include "solvers/solver_callback_handler.hpp"
#include "solvers/solver_runner.hpp"

// Growth rate tolerances
constexpr double E_growth_rate_tolerance = 8e-2;
constexpr double W_growth_rate_tolerance = 5e-2;
// Ignore first few steps to allow rate to stabilise
constexpr int first_check_step = 10;

/**
 * Struct to calculate and record energy and enstrophy growth rates and compare
 * to expected values
 * (see eqns 18-20 https://rnumata.org/research/materials/turb_ws_jan2006.pdf)
 */
struct CalcHWGrowthRates : public NESO::SolverCallback<HW2Din3DSystem> {
  std::vector<double> E;
  std::vector<double> W;
  std::vector<double> E_growth_rate_error;
  std::vector<double> W_growth_rate_error;
  std::vector<double> Gamma_a;
  std::vector<double> Gamma_n;
  void call(HW2Din3DSystem *state) {
    auto md = state->m_diag_growth_rates_recorder;

    E.push_back(md->compute_energy());
    W.push_back(md->compute_enstrophy());
    Gamma_a.push_back(md->compute_Gamma_a());
    Gamma_n.push_back(md->compute_Gamma_n());

    int nsteps = E.size();
    if (nsteps > first_check_step && nsteps > 1) {
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

class HWTest : public NektarSolverTest {
protected:
  void check_growth_rates() {
    CalcHWGrowthRates calc_growth_rates_callback;

    MainFuncType runner = [&](int argc, char **argv) {
      SolverRunner solver_runner(argc, argv);
      auto equation_system = std::dynamic_pointer_cast<HW2Din3DSystem>(
          solver_runner.driver->GetEqu()[0]);

      equation_system->m_solver_callback_handler.register_post_integrate(
          calc_growth_rates_callback);

      solver_runner.execute();
      solver_runner.finalise();
      return 0;
    };

    int ret_code = run(runner);
    EXPECT_EQ(ret_code, 0);

    ASSERT_THAT(calc_growth_rates_callback.E_growth_rate_error,
                testing::Each(testing::Le(E_growth_rate_tolerance)));
    ASSERT_THAT(calc_growth_rates_callback.W_growth_rate_error,
                testing::Each(testing::Le(W_growth_rate_tolerance)));
  }

  std::string get_solver_name() override { return "H3LAPD"; }
};

#endif // H3LAPD_TESTS_H
