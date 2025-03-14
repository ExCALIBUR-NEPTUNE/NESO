#include "SolverIntTest.hpp"

#include <filesystem>
#include <string>

std::filesystem::path
SolverIntTest::get_common_test_resources_dir(std::string solver_name) {
  std::filesystem::path this_dir =
      std::filesystem::path(__FILE__).parent_path();
  return this_dir / solver_name / "common";
}

std::string SolverIntTest::get_run_subdir() {
  return std::string("integration/solvers");
}

std::string SolverIntTest::get_solver_name() {
  return solver_name_from_test_suite_name(
      get_current_test_info()->test_suite_name(), "Test");
}

/**
 * Assume solver test resources are relative to this file at
 * ./<solver_name>/<test_name>/resources
 */
std::filesystem::path
SolverIntTest::get_test_resources_dir(std::string solver_name,
                                      std::string test_name) {
  std::filesystem::path this_dir =
      std::filesystem::path(__FILE__).parent_path();
  return this_dir / solver_name / test_name;
}