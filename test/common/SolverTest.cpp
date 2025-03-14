
#include "SolverTest.hpp"

void SolverTest::cancel_output_redirect() {
  // Restore stdout, stderr buffers
  std::cout.rdbuf(m_orig_stdout_buf);
  m_out_strm.close();
  std::cerr.rdbuf(m_orig_stderr_buf);
  m_err_strm.close();
}

std::vector<std::string> SolverTest::assemble_args() {
  return assemble_default_args();
}

const ::testing::TestInfo *SolverTest::get_current_test_info() {
  return ::testing::UnitTest::GetInstance()->current_test_info();
}

std::vector<std::string> SolverTest::assemble_default_args() {
  std::filesystem::path config_fpath =
      m_test_run_dir / (m_test_name + "_config.xml");
  std::filesystem::path mesh_fpath =
      m_test_run_dir / (m_test_name + "_mesh.xml");
  std::vector<std::string> args;
  args.push_back(m_solver_name);
  args.push_back(std::string(config_fpath));
  args.push_back(std::string(mesh_fpath));
  return args;
}

void SolverTest::make_test_run_dir() {
  if (is_root()) {
    // Remove any previous run dir
    std::filesystem::remove_all(m_test_run_dir);
    // Create dir
    std::filesystem::create_directories(m_test_run_dir.parent_path());

    // Copy in common resources
    if (std::filesystem::exists(m_common_test_res_dir)) {
      std::filesystem::copy(
          m_common_test_res_dir, m_test_run_dir,
          std::filesystem::copy_options::recursive |
              std::filesystem::copy_options::overwrite_existing);
    } else {
      std::cout << "Skipping copy of common resources; no directory at "
                << m_common_test_res_dir << std::endl;
    }

    // Copy in test-specific resources
    if (std::filesystem::exists(m_test_res_dir)) {
      std::filesystem::copy(
          m_test_res_dir, m_test_run_dir,
          std::filesystem::copy_options::recursive |
              std::filesystem::copy_options::overwrite_existing);
    } else {
      std::cout << "Skipping copy of test-specific resources; no directory at "
                << m_test_res_dir << std::endl;
    }
  }
  // Other tasks wait for the dir to be created before continuuing
  int state_at_barrier = MPI_Barrier(MPI_COMM_WORLD);
  if (MPI_SUCCESS != state_at_barrier) {
    std::cout << "MPI_Barrier failed" << std::endl;
  }
}

void SolverTest::print_preamble() {
  std::cout << "Running Nektar solver test [" << get_current_test_info()->name()
            << "]";
  std::cout << " in [" << m_test_run_dir << "]" << std::endl;
  std::cout << "  Args: " << std::endl;
  for (auto ii = 1; ii < m_args.size(); ii++) {
    std::cout << "    " << m_args[ii] << std::endl;
  }
  std::cout << std::endl;
}

void SolverTest::redirect_output_to_file() {
  // Save current buffers
  m_orig_stdout_buf = std::cout.rdbuf();
  m_orig_stderr_buf = std::cerr.rdbuf();
  // Redirect
  std::string stdout_fname = "stdout." + std::to_string(get_rank()) + ".txt";
  std::string stderr_fname = "stderr." + std::to_string(get_rank()) + ".txt";
  m_out_strm.open(m_test_run_dir / stdout_fname);
  std::cout.rdbuf(m_out_strm.rdbuf());
  m_err_strm.open(m_test_run_dir / stderr_fname);
  std::cerr.rdbuf(m_err_strm.rdbuf());
}

int SolverTest::run(MainFuncType func, bool redirect_output) {

  // Create test_dir
  make_test_run_dir();

  // cd to test dir. This is required because paths in the session file are
  // relative to cwd, rather than to the session file itself - the Nektar docs
  // are wrong...
  std::filesystem::path old_dir = std::filesystem::current_path();
  std::filesystem::current_path(m_test_run_dir);

  m_args = assemble_args();

  // Construct argv, argc
  std::vector<char *> cstr_args{};
  for (auto &arg : m_args)
    cstr_args.push_back(&arg.front());
  m_argv = cstr_args.data();
  m_argc = cstr_args.size();

  // Redirect stdout, stderr if requested
  if (redirect_output) {
    redirect_output_to_file();
  }

  print_preamble();

  // Run the solver
  int solver_ret_code = func(m_argc, m_argv);

  // Restore stdout, stderr if they were redirected
  if (redirect_output) {
    cancel_output_redirect();
  }

  // Return to pre-test directory
  std::filesystem::current_path(old_dir);

  return solver_ret_code;
}

std::filesystem::path SolverTest::get_test_run_dir(std::string solver_name,
                                                   std::string test_name) {
  return std::filesystem::temp_directory_path() / "neso-tests" /
         get_run_subdir() / solver_name / test_name;
}

void SolverTest::SetUp() {
  // Set solver name (allowing derived classes to override)
  m_solver_name = get_solver_name();
  // Set test name
  m_test_name = get_current_test_info()->name();

  // Determine test resource locations
  m_common_test_res_dir = get_common_test_resources_dir(m_solver_name);
  m_test_res_dir = get_test_resources_dir(m_solver_name, m_test_name);

  // Set test run location
  m_test_run_dir = get_test_run_dir(m_solver_name, m_test_name);
}