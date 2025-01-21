#include "IntSolverTest.hpp"

#include <filesystem>
#include <string>

std::filesystem::path
IntSolverTest::get_common_test_resources_dir(std::string solver_name) {
  std::filesystem::path this_dir =
      std::filesystem::path(__FILE__).parent_path();
  return this_dir / solver_name / "common";
}

// Asssume solver test resources are in ./<solver_name>/<test_name>/resources
std::filesystem::path
IntSolverTest::get_test_resources_dir(std::string solver_name,
                                      std::string test_name) {
  std::filesystem::path this_dir =
      std::filesystem::path(__FILE__).parent_path();
  return this_dir / solver_name / test_name;
}