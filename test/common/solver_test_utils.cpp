#include "solver_test_utils.hpp"
#include <mpi.h>

// ============================= Helper functions =============================
int get_rank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

bool is_root() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return (size < 2 || get_rank() == 0);
}

// Get solver name from test_suite_name, removing "Test" suffix if necessary
std::string solver_name_from_test_suite_name(const std::string &test_suite_name,
                                             const std::string &suffix) {
  std::string solver_name = test_suite_name;
  if (test_suite_name.size() > suffix.size() &&
      test_suite_name.substr(test_suite_name.size() - suffix.size()) ==
          suffix) {
    solver_name =
        test_suite_name.substr(0, test_suite_name.size() - suffix.size());
  }
  return solver_name;
}

// ============== Member functions for Nektar solver test fixture =============
