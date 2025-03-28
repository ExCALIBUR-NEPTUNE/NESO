#ifndef __NESO_TEST_REGRESSION_EXAMPLES_SOLVERREGTEST_H_
#define __NESO_TEST_REGRESSION_EXAMPLES_SOLVERREGTEST_H_

#include "../../common/SolverTest.hpp"
#include "RegressionData.hpp"

namespace fs = std::filesystem;
class SolverRegTest : public SolverTest {
protected:
  /// Object in which to store reference data for a particular solver example
  RegressionData reg_data;
  /// Default tolerance. May be changed with set_tolerance()
  double tolerance = 1e-10;

  virtual void additional_setup_tasks() override final;

  virtual std::vector<std::string> assemble_args() const override final;

  virtual fs::path get_common_test_resources_dir(
      const std::string &solver_name) const override final;

  std::vector<std::string> get_fpath_args() const;

  std::string get_fpath_arg_str() const;

  virtual std::string get_run_subdir() const override final;

  virtual std::string get_solver_name() const override final;

  virtual fs::path
  get_test_resources_dir(const std::string &solver_name,
                         const std::string &test_name) const override final;

  void run_and_regress();

  void set_tolerance(const double tolerance) { this->tolerance = tolerance; }
};

#endif