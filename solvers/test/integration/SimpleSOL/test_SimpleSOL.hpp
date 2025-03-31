#ifndef __NESOSOLVERS_TESTSIMPLESOL_HPP__
#define __NESOSOLVERS_TESTSIMPLESOL_HPP__
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

#include <FieldUtils/Module.h>
#include <LibUtilities/BasicConst/NektarUnivTypeDefs.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Communication/CommSerial.h>

#include "EquationSystems/SOLWithParticlesSystem.hpp"
#include "solver_test_utils.hpp"
#include "solvers/solver_callback_handler.hpp"
#include "solvers/solver_runner.hpp"

namespace LU = Nektar::LibUtilities;
namespace FU = Nektar::FieldUtils;
namespace PO = boost::program_options;

namespace NESO::Solvers::SimpleSOL {
const int x_idx = 0, rho_idx = 1, vel_idx = 2, T_idx = 3;

class SimpleSOLTest : public NektarSolverTest {
protected:
  void check_mass_conservation(const double &tolerance) {
    if (is_root()) {
      std::vector<std::vector<double>> cons_data = read_conservation_file();
      EXPECT_EQ(cons_data.size(), 4);

      // Require relative error at each step <= tolerance
      const int rel_err_idx = 1;
      ASSERT_THAT(cons_data[rel_err_idx],
                  testing::Each(testing::Le(tolerance)));
    }
  }

  void compare_rho_u_T_profs(const double &tolerance) {
    if (is_root()) {
      std::vector<std::vector<double>> an_data = read_analytic();
      std::vector<std::vector<double>> nektar_data = read_nektar();
      int nvecs = 4; // Data contains s,rho,v_s,T, regardless of mesh dimension
      ASSERT_EQ(an_data.size(), 4);
      ASSERT_EQ(nektar_data.size(), 4);
      ASSERT_EQ(nektar_data[rho_idx].size(), an_data[rho_idx].size());
      ASSERT_EQ(nektar_data[vel_idx].size(), an_data[vel_idx].size());
      ASSERT_EQ(nektar_data[T_idx].size(), an_data[T_idx].size());

      // Require rho, v_s and T profiles to differ (pointwise) by less than
      // <tolerance>
      ASSERT_THAT(nektar_data[rho_idx],
                  testing::Pointwise(DiffLeq(tolerance), an_data[rho_idx]));
      ASSERT_THAT(nektar_data[vel_idx],
                  testing::Pointwise(DiffLeq(tolerance), an_data[vel_idx]));
      ASSERT_THAT(nektar_data[T_idx],
                  testing::Pointwise(DiffLeq(tolerance), an_data[T_idx]));
    }
  }

  std::string get_interp_str(double theta) {
    // Fix s_max = 110, n_pts = 1101 to match analytic data
    int constexpr n_pts = 1101;
    double constexpr s_max = 110.0;
    /* Move interp line away from the boundary by a small amount, otherwise the
     * first point evaluates to zero
     */
    double epsilon = 1e-12;
    double x_min = 0.0;
    double y_min = 0.0 + epsilon;
    double x_max = s_max * cos(theta) - epsilon;
    double y_max = s_max * sin(theta);
    std::stringstream ss;
    ss << n_pts << "," << x_min << "," << y_min << "," << x_max << "," << y_max;
    return ss.str();
  }

  std::vector<std::vector<double>> read_analytic() {
    return read_csv("analytic_sigma2.csv", 4);
  }

  std::vector<std::vector<double>> read_conservation_file() {
    return read_csv("mass_recording.csv", 4);
  }

  std::vector<std::vector<double>> read_csv(std::string fname, int ncols) {
    std::ifstream an_file;
    an_file.open(m_test_run_dir / fname);

    // Header
    std::string header_str;
    std::getline(an_file, header_str);

    // Values
    std::vector<std::vector<double>> vals(ncols);
    std::string line;
    while (std::getline(an_file, line)) {
      std::vector<std::string> str_vals;
      boost::algorithm::split(str_vals, line, boost::algorithm::is_any_of(","));
      for (auto icol = 0; icol < str_vals.size(); icol++) {
        vals[icol].push_back(std::stod(str_vals[icol]));
      }
    }

    an_file.close();
    return vals;
  }

  std::vector<std::vector<double>> read_nektar() {
    FU::FieldSharedPtr f = std::make_shared<FU::Field>();
    // Set up a (serial) communicator
    f->m_comm = LU::GetCommFactory().CreateInstance("Serial", m_argc, m_argv);

    // Several module.process() funcs take a variable map but don't do anything
    // with it; create a dummy map to make them work
    po::variables_map dummy;

    // Read config, mesh from xml
    FU::ModuleKey readXmlKey =
        std::make_pair(FU::ModuleType::eInputModule, "xml");
    FU::ModuleSharedPtr readXmlMod =
        FU::GetModuleFactory().CreateInstance(readXmlKey, f);
    readXmlMod->AddFile("xml", std::string(m_args[1]));
    readXmlMod->RegisterConfig("infile", m_args[1]);
    readXmlMod->AddFile("xml", std::string(m_args[2]));
    readXmlMod->RegisterConfig("infile", m_args[2]);
    readXmlMod->Process(dummy);

    // Interpolate from .fld file
    std::string fld_fpath = f->m_session->GetSessionName() + ".fld";

    Nektar::NekDouble THETA;
    f->m_session->LoadParameter("theta", THETA, 0.0);
    std::string line_interp_str = get_interp_str(THETA);
    FU::ModuleKey interpModKey =
        std::make_pair(FU::ModuleType::eProcessModule, "interppoints");
    FU::ModuleSharedPtr interpMod =
        FU::GetModuleFactory().CreateInstance(interpModKey, f);
    interpMod->RegisterConfig("fromxml", std::string(m_args[1]) + "," +
                                             std::string(m_args[2]));
    interpMod->RegisterConfig("fromfld", fld_fpath);
    interpMod->RegisterConfig("line", line_interp_str);
    // All other config options must be set, otherwise exceptions are thrown
    interpMod->SetDefaults();
    interpMod->Process(dummy);

    // Retrieve values from Nektar Field object
    int ndims = f->m_graph->GetMeshDimension();
    int nek_x_idx = 0;
    // N.B. The format of 'line_interp_str' (n, minx, miny, maxx,maxy) means
    // that the interpolation always returns coords in the first 2 indices,
    // regardless of ndims; hence rho is in idx 2
    int nek_rho_idx = 2;
    int nek_rhou_idx = nek_rho_idx + 1;
    int nek_E_idx = nek_rhou_idx + ndims;
    Nek1DArr x = f->m_fieldPts->GetPts(nek_x_idx);
    Nek1DArr rho = f->m_fieldPts->GetPts(nek_rho_idx);
    Nek1DArr rhou = f->m_fieldPts->GetPts(nek_rhou_idx);
    Nek1DArr rhov;
    if (ndims == 2) {
      int nek_rhov_idx = nek_rhou_idx + 1;
      rhov = f->m_fieldPts->GetPts(nek_rhov_idx);
    }

    Nek1DArr E = f->m_fieldPts->GetPts(nek_E_idx);

    // Get param values from config
    Nektar::NekDouble GAMMA;
    Nektar::NekDouble GASCONSTANT;
    f->m_session->LoadParameter("Gamma", GAMMA);
    f->m_session->LoadParameter("GasConstant", GASCONSTANT);

    // Regardlesss of the mesh dimension, only return x, rho, v_s, T (vals
    // size=4)
    std::vector<std::vector<double>> vals(4);

    // Put field values into a vector<vector<double>, converting momentum to
    // velocity and energy to temperature
    for (auto ii = 0; ii < f->m_fieldPts->GetNpoints(); ii++) {
      vals[x_idx].push_back(x[ii]);
      vals[rho_idx].push_back(rho[ii]);
      double vel = cos(THETA) * rhou[ii] / rho[ii];
      if (ndims == 2) {
        vel += sin(THETA) * rhov[ii] / rho[ii];
      }
      vals[vel_idx].push_back(vel);
      // E => T using ideal gas law
      vals[T_idx].push_back((E[ii] - (rho[ii] * vel * vel) / 2) / rho[ii] *
                            (GAMMA - 1) / GASCONSTANT);
    }
    return vals;
  }
};

struct SOLWithParticlesMassConservationPre
    : public NESO::SolverCallback<SOLWithParticlesSystem> {
  void call(SOLWithParticlesSystem *state) {
    state->diag_mass_recording->compute_initial_fluid_mass();
  }
};

struct SOLWithParticlesMassConservationPost
    : public NESO::SolverCallback<SOLWithParticlesSystem> {
  std::vector<double> mass_error;
  void call(SOLWithParticlesSystem *state) {
    auto md = state->diag_mass_recording;
    const double mass_particles = md->compute_particle_mass();
    const double mass_fluid = md->compute_fluid_mass();
    const double mass_total = mass_particles + mass_fluid;
    const double mass_added = md->compute_total_added_mass();
    const double correct_total = mass_added + md->get_initial_mass();
    this->mass_error.push_back(std::fabs(correct_total - mass_total) /
                               std::fabs(correct_total));
  }
};
} // namespace NESO::Solvers::SimpleSOL

#endif // __NESOSOLVERS_TESTSIMPLESOL_HPP__
