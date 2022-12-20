#ifndef SIMPLESOL_TESTS_COMMON
#define SIMPLESOL_TESTS_COMMON

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <gtest/gtest.h>

#include <FieldUtils/Module.h>
#include <LibUtilities/BasicConst/NektarUnivTypeDefs.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Communication/CommSerial.h>

#include "SimpleSOL.h"
#include "solver_test_utils.h"

namespace LU = Nektar::LibUtilities;
namespace FU = Nektar::FieldUtils;
namespace PO = boost::program_options;

const int x_idx = 0, rho_idx = 1, u_idx = 2, T_idx = 3;

class SimpleSOLTest : public NektarSolverTest {
protected:
  void compare_rho_u_T_profs(const double &tolerance) {
    std::vector<std::vector<double>> an_data = read_analytic();
    std::vector<std::vector<double>> nektar_data = read_nektar();
    int nvecs = 4; // Data contains x,rho,u,T, regardless of mesh dimension
    ASSERT_EQ(an_data.size(), 4);
    ASSERT_EQ(nektar_data.size(), 4);
    ASSERT_EQ(nektar_data[rho_idx].size(), an_data[rho_idx].size());
    ASSERT_EQ(nektar_data[u_idx].size(), an_data[u_idx].size());
    ASSERT_EQ(nektar_data[T_idx].size(), an_data[T_idx].size());

    // Require rho, u and T profiles to differ (pointwise) by less than <tolerance>
    ASSERT_THAT(nektar_data[rho_idx], testing::Pointwise(DiffLeq(tolerance), an_data[rho_idx]));
    ASSERT_THAT(nektar_data[u_idx], testing::Pointwise(DiffLeq(tolerance), an_data[u_idx]));
    ASSERT_THAT(nektar_data[T_idx], testing::Pointwise(DiffLeq(tolerance), an_data[T_idx]));
  }

  std::vector<std::vector<double>> read_analytic() {
    std::ifstream an_file;
    an_file.open(m_test_run_dir / "analytic_sigma2.csv");

    // Header
    std::string header_str;
    std::getline(an_file, header_str);

    // Values
    std::vector<std::vector<double>> vals(4);
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

    // Several module.process() funcs take a variable map but don't do anything with it; create a
    // dummy map to make them work
    po::variables_map dummy;

    // Read config, mesh from xml
    FU::ModuleKey readXmlKey = std::make_pair(FU::ModuleType::eInputModule, "xml");
    FU::ModuleSharedPtr readXmlMod = FU::GetModuleFactory().CreateInstance(readXmlKey, f);
    readXmlMod->AddFile("xml", std::string(m_args[1]));
    readXmlMod->RegisterConfig("infile", m_args[1]);
    readXmlMod->AddFile("xml", std::string(m_args[2]));
    readXmlMod->RegisterConfig("infile", m_args[2]);
    readXmlMod->Process(dummy);

    // Interpolate from .fld file
    std::string fld_fpath = f->m_session->GetSessionName() + ".fld";
    std::string line_interp_str = "1101,0,0,110,0";
    FU::ModuleKey interpModKey = std::make_pair(FU::ModuleType::eProcessModule, "interppoints");
    FU::ModuleSharedPtr interpMod = FU::GetModuleFactory().CreateInstance(interpModKey, f);
    interpMod->RegisterConfig("fromxml", std::string(m_args[1]) + "," + std::string(m_args[2]));
    interpMod->RegisterConfig("fromfld", fld_fpath);
    interpMod->RegisterConfig("line", line_interp_str);
    // All other config options must be set, otherwise exceptions are thrown
    interpMod->SetDefaults();
    interpMod->Process(dummy);

    // Retrieve values from Nektar Field object
    int ndims = f->m_graph->GetMeshDimension();
    int nek_x_idx = 0;
    // N.B. The format of 'line_interp_str' (n, minx, miny, maxx,maxy) means that the interpolation
    // always returns coords in the first 2 indices, regardless of ndims; hence rho is in idx 2
    int nek_rho_idx = 2;
    int nek_rhou_idx = nek_rho_idx + 1;
    int nek_E_idx = nek_rhou_idx + ndims;
    Nek1DArr x = f->m_fieldPts->GetPts(nek_x_idx);
    Nek1DArr rho = f->m_fieldPts->GetPts(nek_rho_idx);
    Nek1DArr rhou = f->m_fieldPts->GetPts(nek_rhou_idx);
    Nek1DArr E = f->m_fieldPts->GetPts(nek_E_idx);

    // Get param values from config
    Nektar::NekDouble GAMMA;
    Nektar::NekDouble GASCONSTANT;
    f->m_session->LoadParameter("Gamma", GAMMA);
    f->m_session->LoadParameter("GasConstant", GASCONSTANT);

    // Regardlesss of the mesh dimension, only return x, rho, u, T (vals size=4)
    std::vector<std::vector<double>> vals(4);

    // Put field values into a vector<vector<double>, converting rho*u => u and E => T
    for (auto ii = 0; ii < f->m_fieldPts->GetNpoints(); ii++) {
      vals[x_idx].push_back(x[ii]);
      vals[rho_idx].push_back(rho[ii]);
      vals[u_idx].push_back(rhou[ii] / rho[ii]);
      // E => T using ideal gas law
      vals[T_idx].push_back((E[ii] - (rhou[ii] * rhou[ii] / rho[ii]) / 2) / rho[ii] * (GAMMA - 1) /
                            GASCONSTANT);
    }
    return vals;
  }
};

#endif // SIMPLESOL_TESTS_COMMON