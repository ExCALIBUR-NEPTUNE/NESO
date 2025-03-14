#ifndef __NESO_TEST_REGRESSION_EXAMPLES_SOLVERREGTEST_H_
#define __NESO_TEST_REGRESSION_EXAMPLES_SOLVERREGTEST_H_

#include "../../common/SolverTest.hpp"
#include "RegressionData.hpp"

namespace fs = std::filesystem;
class SolverRegTest : public SolverTest {
protected:
  RegressionData reg_data;
  /// Default tolerance. May be changed with set_tolerance()
  double tolerance = 1e-12;

  virtual void additional_setup_tasks() override final;

  virtual std::vector<std::string> assemble_args() override final;

  virtual fs::path
  get_common_test_resources_dir(std::string solver_name) override final;

  std::vector<std::string> get_fpath_args();

  std::string get_fpath_arg_str();

  virtual std::string get_run_subdir() override final;

  virtual std::string get_solver_name() override final;

  virtual fs::path get_test_resources_dir(std::string solver_name,
                                          std::string test_name) override final;

  void run_and_regress();

  void set_tolerance(const double tolerance) { this->tolerance = tolerance; }
};

#endif