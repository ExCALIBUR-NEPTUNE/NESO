#ifndef __TEST_INTEGRATION_INTSOLVERTEST_H_
#define __TEST_INTEGRATION_INTSOLVERTEST_H_

#include "../common/SolverTest.hpp"

class SolverIntTest : public SolverTest {
protected:
  virtual std::filesystem::path get_common_test_resources_dir(
      const std::string &solver_name) const override final;

  virtual std::string get_run_subdir() const override final;

  virtual std::string get_solver_name() const override;

  virtual std::filesystem::path
  get_test_resources_dir(const std::string &solver_name,
                         const std::string &test_name) const override final;
};

#endif