#ifndef __TEST_COMMON_SOLVERTEST_H_
#define __TEST_COMMON_SOLVERTEST_H_

#include <filesystem>
#include <fstream>
#include <gmock/gmock.h>

#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include "solver_test_utils.hpp"

typedef Nektar::Array<Nektar::OneD, Nektar::NekDouble> Nek1DArr;

class SolverTest : public ::testing::Test {
protected:
  // File streams and buffers for stdout, stderr, used if redirecting output
  std::ofstream m_err_strm, m_out_strm;
  std::streambuf *m_orig_stdout_buf;
  std::streambuf *m_orig_stderr_buf;

  // Directory paths
  std::filesystem::path m_test_res_dir;
  std::filesystem::path m_test_run_dir;
  std::filesystem::path m_common_test_res_dir;

  // Store solver, test name for convenience
  std::string m_solver_name;
  std::string m_test_name;

  // Run args
  std::vector<std::string> m_args;
  char **m_argv;
  int m_argc;

  // Subclasses can override this in order to add additional tasks to SetUp()
  virtual void additional_setup_tasks(){};

  // Construct argument vector (including paths to config, mesh xmls)
  virtual std::vector<std::string> assemble_args() const;

  // Construct argument vector based on the test name and run directory
  std::vector<std::string> assemble_default_args() const;

  // Cancel redirect initiated with redirect_output_to_file()
  void cancel_output_redirect();

  virtual std::filesystem::path
  get_common_test_resources_dir(const std::string &solver_name) const = 0;

  // Convenience function to get current test info
  const ::testing::TestInfo *get_current_test_info() const;

  // Subclasses must override this to define test run locations relative to
  // std::filesystem::temp_directory_path() / "neso-tests"
  virtual std::string get_run_subdir() const = 0;

  // Subclasses must override to set solver_name
  virtual std::string get_solver_name() const = 0;

  virtual std::filesystem::path
  get_test_resources_dir(const std::string &solver_name,
                         const std::string &test_name) const = 0;

  std::filesystem::path get_test_run_dir(const std::string &solver_name,
                                         const std::string &test_name) const;

  // Create a temporary directory to run the test in and copy in required
  // resources
  void make_test_run_dir() const;

  void print_preamble() const;

  // Redirect stdout, stderr to files in test run dir
  void redirect_output_to_file();

  // Run a solver
  int run(MainFuncType func, bool redirect_output = true);

  /// Prevent further overrides of SetUp to ensure
  void SetUp() override final;
};

#endif