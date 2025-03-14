#ifndef __NESO_TEST_REGRESSION_EXAMPLES_SOLVERREGTEST_H_
#define __NESO_TEST_REGRESSION_EXAMPLES_SOLVERREGTEST_H_

#include "../../common/SolverTest.hpp"
#include "RegressionData.hpp"

class SolverRegTest : public SolverTest {
protected:
  virtual void additional_setup_tasks() override final;

  virtual std::vector<std::string> assemble_args() override final;

  virtual std::filesystem::path
  get_common_test_resources_dir(std::string solver_name) override final;

  virtual std::string get_run_subdir() override final;

  virtual std::string get_solver_name() override final;

  virtual std::filesystem::path
  get_test_resources_dir(std::string solver_name,
                         std::string test_name) override final;

private:
  RegressionData reg_data;
};

#endif