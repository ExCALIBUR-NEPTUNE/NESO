#include "SolverRegTest.hpp"
#include <filesystem>
#include <string>

#include <FieldUtils/Module.h>
#include <LibUtilities/Communication/CommSerial.h>
#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <hdf5.h>
#include <solvers/solver_callback_handler.hpp>
#include <solvers/solver_runner.hpp>

namespace FU = Nektar::FieldUtils;
using NekDouble = Nektar::NekDouble;

// ============================= Helper functions =============================
/**
 * @brief Determine whether an argument is a filename
 *
 * @param arg a solver argument string
 * @return true if arg should be treated as a filename, false otherwise
 */
bool is_file_arg(const std::string &arg) {
  return fs::path(arg).has_extension();
}

/**
 * @brief Parse an example command template from examples dir and extract solver
 * arguments
 *
 * @param[in] path Path to the template file
 * @param[out] args A vector of the arguments extracted
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
//=============================================================================

/**
 * @brief Read regression data before test runs.
 *
 */
void SolverRegTest::additional_setup_tasks() {
  // Read regression data. Fail test on error.
  std::string reg_data_fname = this->test_name + ".regression_data.h5";
  fs::path reg_data_path =
      fs::path(__FILE__).parent_path().parent_path().parent_path() /
      get_run_subdir() / this->solver_name / reg_data_fname;
  this->reg_data.read(reg_data_path);
  if (this->reg_data.err_state != 0) {
    FAIL() << this->reg_data;
  }
}

/**
 * @brief Extract example arguments from command template, adjust input
 * filepaths and add args to ensure number of steps matches regression data.
 *
 * @return std::vector<std::string> vector of argument strings
 */
std::vector<std::string> SolverRegTest::assemble_args() const {
  std::vector<std::string> args;

  // Retrieve args (config, mesh filenames etc.) from the command template
  fs::path template_path(this->test_res_dir / "run_cmd_template.txt");
  parse_cmd_template(template_path, args);

  // Assume args with extensions are filenames - prepend test_run dir to them
  for (auto ii = 0; ii < args.size(); ii++) {
    if (is_file_arg(args[ii])) {
      args[ii] = fs::path(this->test_run_dir / args[ii]).string();
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
  args.insert(args.begin(), this->solver_name);
  return args;
}

fs::path SolverRegTest::get_common_test_resources_dir(
    const std::string &solver_name) const {
  fs::path this_dir = fs::path(__FILE__).parent_path();
  return this_dir / solver_name / "common";
}

std::string SolverRegTest::get_fpath_arg_str() const {
  std::stringstream ss;
  std::vector<std::string> fpath_args = get_fpath_args();
  ss << fpath_args[0];
  for (auto ii = 1; ii < fpath_args.size(); ii++) {
    ss << ", " << fpath_args[ii];
  }
  return ss.str();
}

/**
 * @brief Filter the argument list to isolate filepaths
 *
 * @return std::vector<std::string> a vector of all args identified as filepaths
 */
std::vector<std::string> SolverRegTest::get_fpath_args() const {
  // Filter args with is_file_arg
  std::vector<std::string> fpath_args;
  std::copy_if(this->args.begin(), this->args.end(),
               std::back_inserter(fpath_args),
               [](const std::string &p) { return is_file_arg(p); });
  return fpath_args;
}

/**
 * @brief Specify top-level directory in which all solver example regression
 * tests will run.
 *
 * @return std::string relative path to run directory
 */
std::string SolverRegTest::get_run_subdir() const {
  return std::string("regression/examples");
}

/**
 * @brief Specify mapping from test name to solver name.
 *
 * @return std::string solver name to be associated with this test
 */
std::string SolverRegTest::get_solver_name() const {
  return solver_name_from_test_suite_name(
      get_current_test_info()->test_suite_name(), "RegTest");
}

/**
 * @brief Specify location in which to look for solver test resources
 *
 * @param solver_name name of the Solver being tested
 * @param test_name name of the test
 * @return fs::path path to the test resources dir
 */
fs::path
SolverRegTest::get_test_resources_dir(const std::string &solver_name,
                                      const std::string &test_name) const {
  // <repo_root>/examples/<solver_name>/<test_name>
  fs::path this_dir = fs::path(__FILE__).parent_path();
  return this_dir / "../../../examples" / solver_name / test_name;
}

/**
 * @brief Function to run a solver example, extract field values from the output
 * file and compare them to the regression data.
 *
 */
void SolverRegTest::run_and_regress() {
  // Run solver
  MainFuncType runner = [&](int argc, char **argv) {
    SolverRunner solver_runner(argc, argv);
    solver_runner.execute();
    solver_runner.finalise();
    return 0;
  };

  int ret_code = run(runner);
  ASSERT_EQ(ret_code, 0);

  // Read .fld file and create equispaced points
  FU::FieldSharedPtr f = std::make_shared<FU::Field>();
  // Set up a (serial) communicator
  f->m_comm =
      LU::GetCommFactory().CreateInstance("Serial", this->argc, this->argv);

  // Dummy map required for module.process()
  po::variables_map empty_var_map;

  // Read config, mesh from xml
  FU::ModuleKey readXmlKey =
      std::make_pair(FU::ModuleType::eInputModule, "xml");
  FU::ModuleSharedPtr readXmlMod =
      FU::GetModuleFactory().CreateInstance(readXmlKey, f);
  std::vector<std::string> str_args = get_fpath_args();
  for (std::string &arg : str_args) {
    readXmlMod->AddFile("xml", arg);
    readXmlMod->RegisterConfig("infile", arg);
  }
  readXmlMod->Process(empty_var_map);

  // Read fld
  std::string fld_fpath = f->m_session->GetSessionName() + ".fld";
  FU::ModuleKey readFldKey =
      std::make_pair(FU::ModuleType::eInputModule, "fld");
  FU::ModuleSharedPtr readFldMod =
      FU::GetModuleFactory().CreateInstance(readFldKey, f);
  readFldMod->RegisterConfig("infile", fld_fpath);
  readFldMod->Process(empty_var_map);

  // Generate equi-spaced points
  FU::ModuleKey equiPtsModKey =
      std::make_pair(FU::ModuleType::eProcessModule, "equispacedoutput");
  FU::ModuleSharedPtr equiPtsMod =
      FU::GetModuleFactory().CreateInstance(equiPtsModKey, f);
  equiPtsMod->Process(empty_var_map);

  // Copy equispaced pts into a map to simplify comparison with regression data
  std::map<std::string, std::vector<NekDouble>> run_results;
  std::vector<std::string> fld_names = f->m_fieldPts->GetFieldNames();
  int ndims = f->m_graph->GetMeshDimension();
  for (int ifld = 0; ifld < fld_names.size(); ifld++) {
    Nektar::Array<Nektar::OneD, Nektar::NekDouble> fld_vals =
        f->m_fieldPts->GetPts(ifld + ndims);
    run_results[fld_names[ifld]] = std::vector<NekDouble>(fld_vals.size());
    for (int ipt = 0; ipt < fld_vals.size(); ipt++) {
      run_results[fld_names[ifld]][ipt] = fld_vals[ipt];
    }
  }

  std::function<double(const double &a, const double &b)> calc_abs_diff =
      [](const double &a, const double &b) { return std::abs(a - b); };

  // Compare result to regression data for each field
  for (auto &[fld_name, result_vals] : run_results) {
    int reg_dsize = this->reg_data.dsets[fld_name].size();
    int test_dsize = result_vals.size();
    ASSERT_THAT(test_dsize, reg_dsize)
        << "Test data size (" << test_dsize
        << ") doesn't match regression data size (" << reg_dsize << ")"
        << std::endl;
    std::vector<double> diff(test_dsize);
    std::transform(result_vals.begin(), result_vals.end(),
                   this->reg_data.dsets[fld_name].begin(), diff.begin(),
                   calc_abs_diff);

    auto max_diff_elt = std::max_element(diff.begin(), diff.end());
    auto max_diff_idx = std::distance(diff.begin(), max_diff_elt);
    // Each equi-spaced point must match regression data to within tolerance
    ASSERT_THAT(diff, testing::Each(testing::Le(this->tolerance)))
        << std::endl
        << "Max " << fld_name << " difference was " << *max_diff_elt << " ("
        << result_vals[max_diff_idx] << " in test, "
        << this->reg_data.dsets[fld_name][max_diff_idx]
        << " in regression data)" << std::endl;
  }
}