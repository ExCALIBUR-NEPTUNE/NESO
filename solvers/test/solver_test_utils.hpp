#ifndef __NESOSOLVERS_SOLVERTESTUTILS_HPP__
#define __NESOSOLVERS_SOLVERTESTUTILS_HPP__

#include <filesystem>
#include <fstream>
#include <gmock/gmock.h>
#include <string>

#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include "solvers/solver_runner.hpp"

// Types
typedef std::function<int(int argc, char *argv[])> MainFuncType;
typedef Nektar::Array<Nektar::OneD, Nektar::NekDouble> Nek1DArr;

// ================================ Test macros ===============================
/**
 * Define a matcher for use in gmock's 'Pointwise'
 * Returns true if difference between elements is less-than-or-equal to <diff>
 * e.g. , EXPECT_THAT(container1, testing::Pointwise(DiffLeq(1e-6),
 * container2));
 */
MATCHER_P(DiffLeq, diff, "") {
  return std::abs(std::get<0>(arg) - std::get<1>(arg)) <= diff;
}

// ============================= Helper functions =============================
std::vector<std::string> get_default_args(std::string test_suite_name,
                                          std::string test_name);
int get_rank();
std::filesystem::path get_test_run_dir(std::string solver_name,
                                       std::string test_name);
bool is_root();
std::string solver_name_from_test_suite_name(std::string test_suite_name);

// ================ Base test fixture class for Nektar solvers ================
class NektarSolverTest : public ::testing::Test {
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

  // Cancel redirect initiated with redirect_output_to_file()
  void cancel_output_redirect();

  // Convenience function to get current test info
  const ::testing::TestInfo *get_current_test_info();

  // Construct an argument vector (including paths to config, mesh xmls) based
  // on the test name
  std::vector<std::string> get_default_args();

  // Allow derived classes to override solver_name
  virtual std::string get_solver_name();

  // Create a temporary directory to run the test in and copy in required
  // resources
  void make_test_run_dir();

  void print_preamble();

  // Redirect stdout, stderr to files in test run dir
  void redirect_output_to_file();

  // Run a solver, passing in different args to those returned by
  // get_default_args(), if required
  int run(MainFuncType solver_entrypoint = run_solver,
          std::vector<std::string> args = std::vector<std::string>(),
          bool redirect_output = true);

  void SetUp() override;
};
#endif // ifndef __NESOSOLVERS_SOLVERTESTUTILS_HPP__