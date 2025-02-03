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

bool is_file_arg(const std::string &arg) {
  return fs::path(arg).has_extension();
}

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

std::vector<std::string> SolverRegTest::assemble_args() const {
  std::vector<std::string> args;

  // Retrieve args (config, mesh filenames etc.) from the command template
  fs::path template_path(m_test_res_dir / "run_cmd_template.txt");
  parse_cmd_template(template_path, args);

  // Assume args with extensions are filenames - prepend test_run dir to them
  for (auto ii = 0; ii < args.size(); ii++) {
    if (is_file_arg(args[ii])) {
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

std::vector<std::string> SolverRegTest::get_fpath_args() const {
  // Filter m_args with is_file_arg
  std::vector<std::string> fpath_args;
  std::copy_if(m_args.begin(), m_args.end(), std::back_inserter(fpath_args),
               [](const std::string &p) { return is_file_arg(p); });
  return fpath_args;
}

std::string SolverRegTest::get_run_subdir() const {
  return std::string("regression/examples");
}

std::string SolverRegTest::get_solver_name() const {
  return solver_name_from_test_suite_name(
      get_current_test_info()->test_suite_name(), "RegTest");
}

// Look for solver test resources in
// <repo_root>/examples/<solver_name>/<test_name>
fs::path
SolverRegTest::get_test_resources_dir(const std::string &solver_name,
                                      const std::string &test_name) const {
  fs::path this_dir = fs::path(__FILE__).parent_path();
  return this_dir / "../../../examples" / solver_name / test_name;
}

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
  f->m_comm = LU::GetCommFactory().CreateInstance("Serial", m_argc, m_argv);

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

  std::function<int(const double &a, const double &b)> abs_diff =
      [](const double &a, const double &b) { return std::abs(a - b); };

  // Compare result to regression data for each field
  for (auto &[fld_name, result_vals] : run_results) {
    std::vector<double> diff(result_vals.size());
    std::transform(result_vals.begin(), result_vals.end(),
                   this->reg_data.dsets[fld_name].begin(), diff.begin(),
                   abs_diff);
    // Each equi-spaced point must match regression data to within tolerance
    ASSERT_THAT(diff, testing::Each(testing::Le(this->tolerance)));
  }
}