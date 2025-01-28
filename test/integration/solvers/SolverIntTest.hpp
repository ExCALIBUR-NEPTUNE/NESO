#ifndef __TEST_INTEGRATION_INTSOLVERTEST_H_
#define __TEST_INTEGRATION_INTSOLVERTEST_H_

#include "../../common/SolverTest.hpp"

class SolverIntTest : public SolverTest {
protected:
  virtual std::filesystem::path
  get_common_test_resources_dir(std::string solver_name) override final;

  virtual std::string get_run_subdir() override final;

  virtual std::string get_solver_name() override;

  virtual std::filesystem::path
  get_test_resources_dir(std::string solver_name,
                         std::string test_name) override final;
};

#endif