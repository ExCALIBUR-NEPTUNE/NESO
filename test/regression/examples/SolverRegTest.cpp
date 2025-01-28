#include "SolverRegTest.hpp"
#include <filesystem>
#include <string>

#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <hdf5.h>

namespace fs = std::filesystem;

/**
 * @brief Parse command template from examples dir
 */
void parse_cmd_template(const fs::path &path, std::vector<std::string> &args) {
  if (fs::exists(path)) {
    std::ifstream in_strm;
    in_strm.open(path);
    std::string line;
    std::getline(in_strm, line);
    const std::string solver_tag("<SOLVER_EXEC> ");
    std::size_t arg_start_pos = line.find(solver_tag);
    if (arg_start_pos >= 0) {
      std::string arg_str = line.substr(arg_start_pos + solver_tag.size());
      boost::algorithm::split(args, arg_str, boost::algorithm::is_space());
    } else {
      FAIL() << "Solver tag '" << solver_tag
             << "' not found in command template at " << path.string();
    }
  } else {
    FAIL() << "Expected an example command template at " << path.string();
  }
}

void SolverRegTest::additional_setup_tasks() {
  // Read regression data. Fail test on error.
  fs::path reg_data_file =
      fs::path(__FILE__).parent_path().parent_path().parent_path() /
      get_run_subdir() / m_solver_name / m_test_name / "regression_data.h5";
  this->reg_data.read(reg_data_file);
  if (this->reg_data.err_state != 0) {
    FAIL() << this->reg_data;
  }
}

std::vector<std::string> SolverRegTest::assemble_args() {
  std::vector<std::string> args;

  // Retrieve args (config, mesh filenames etc.) from the command template
  fs::path template_path(m_test_res_dir / "run_cmd_template.txt");
  parse_cmd_template(template_path, args);

  // Assume args with extensions are filenames - prepend test_run dir to them
  for (auto ii = 0; ii < args.size(); ii++) {
    if (fs::path(args[ii]).has_extension()) {
      args[ii] = fs::path(m_test_run_dir / args[ii]).string();
    }
  }

  /* Regression data stores the number of steps taken when it was generated.
   * Enforce the same sim length in the test run via a command line param.
   */
  std::stringstream num_steps_ss, num_chk_steps_ss;
  num_steps_ss << "NumSteps=" << reg_data.nsteps;
  num_chk_steps_ss << "IO_CheckSteps=" << (reg_data.nsteps + 1);
  args.push_back("--parameter");
  args.push_back(num_steps_ss.str());
  args.push_back("--parameter");
  args.push_back(num_chk_steps_ss.str());

  // Insert exec label (solver name) as the first arg
  args.insert(args.begin(), m_solver_name);
  return args;
}

fs::path SolverRegTest::get_common_test_resources_dir(std::string solver_name) {
  fs::path this_dir = fs::path(__FILE__).parent_path();
  return this_dir / solver_name / "common";
}

std::string SolverRegTest::get_run_subdir() {
  return std::string("regression/examples");
}

std::string SolverRegTest::get_solver_name() {
  return solver_name_from_test_suite_name(
      get_current_test_info()->test_suite_name(), "RegTest");
}

// Look for solver test resources in
// <repo_root>/examples/<solver_name>/<test_name>
fs::path SolverRegTest::get_test_resources_dir(std::string solver_name,
                                               std::string test_name) {
  fs::path this_dir = fs::path(__FILE__).parent_path();
  return this_dir / "../../../examples" / solver_name / test_name;
}