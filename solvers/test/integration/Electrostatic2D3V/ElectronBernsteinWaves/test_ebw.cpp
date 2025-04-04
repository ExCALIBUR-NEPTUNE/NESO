#include <gtest/gtest.h>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <boost/math/statistics/linear_regression.hpp>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>
#include <SpatialDomains/MeshGraphIO.h>

#include "../../../../../solvers/Electrostatic2D3V/ElectrostaticElectronBernsteinWaves2D3V.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using boost::math::statistics::simple_ordinary_least_squares;
namespace ES2D3V = NESO::Solvers::Electrostatic2D3V;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

#ifndef FIELD_TYPE
#define FIELD_TYPE ContField
#endif

TEST(Electrostatic2D3V, ElectrostaticElectronBernsteinWaves) {
  std::cout << "Running Electrostatic2D3V.ElectrostaticElectronBernsteinWaves"
            << std::endl;
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  string vDriverModule;
  DriverSharedPtr drv;

  int argc = 3;
  char *argv[3];

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path conditions_file = source_dir / "ebw_conditions.xml";
  std::filesystem::path mesh_file = source_dir / "ebw_mesh.xml";

  copy_to_cstring(std::string("test_ebw"), &argv[0]);
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraphIO::Read(session);

  // Create driver
  session->LoadSolverInfo("Driver", vDriverModule, "Standard");
  drv = GetDriverFactory().CreateInstance(vDriverModule, session, graph);

  auto electrostatic_ebw_2d3v = std::make_shared<
      ES2D3V::ElectrostaticElectronBernsteinWaves2D3V<FIELD_TYPE>>(session,
                                                                   graph, drv);

  // space to store energy
  std::vector<double> potential_energy;
  std::vector<double> total_energy;
  std::vector<double> time_steps;
  const int num_time_steps = electrostatic_ebw_2d3v->num_time_steps;
  potential_energy.reserve(num_time_steps);
  total_energy.reserve(num_time_steps);
  time_steps.reserve(num_time_steps);

  // call back function that executes the energy computation loops and records
  // the outputs
  std::function<void(
      ES2D3V::ElectrostaticElectronBernsteinWaves2D3V<FIELD_TYPE> *)>
      collect_energy =
          [&](ES2D3V::ElectrostaticElectronBernsteinWaves2D3V<FIELD_TYPE>
                  *state) {
            const int time_step = state->time_step;
            if (time_step % 20 == 0) {
              state->potential_energy->compute();
              state->kinetic_energy->compute();
              const double pe = state->potential_energy->energy;
              const double ke = state->kinetic_energy->energy;
              const double te = pe + ke;
              potential_energy.push_back(std::log(pe));
              total_energy.push_back(te);
              time_steps.push_back(time_step * state->charged_particles->dt);
            }
          };
  electrostatic_ebw_2d3v->push_callback(collect_energy);

  // run the simulation
  electrostatic_ebw_2d3v->run();

  electrostatic_ebw_2d3v->finalise();
  session->Finalise();

  // check the energy conservation over the whole simulation
  ASSERT_NEAR(1,
              total_energy[total_energy.size() - 1] / std::abs(total_energy[0]),
              1.0e-2);

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
