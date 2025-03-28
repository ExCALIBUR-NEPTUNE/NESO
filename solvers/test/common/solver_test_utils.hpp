#ifndef TEST_COMMON_SOLVERUTILS
#define TEST_COMMON_SOLVERUTILS

#include <filesystem>
#include <gmock/gmock.h>
#include <string>

// Types
typedef std::function<int(int argc, char *argv[])> MainFuncType;

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
int get_rank();
std::filesystem::path get_test_run_dir(std::string solver_name,
                                       std::string test_name);
bool is_root();

std::string solver_name_from_test_suite_name(const std::string &test_suite_name,
                                             const std::string &suffix);

// ================ Base test fixture class for Nektar solvers ================

#endif // ifndef TEST_COMMON_SOLVERUTILS