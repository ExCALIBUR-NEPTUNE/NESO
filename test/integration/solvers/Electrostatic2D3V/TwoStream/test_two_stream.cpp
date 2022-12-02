#include <gtest/gtest.h>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <boost/math/statistics/linear_regression.hpp>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>

#include "../../../../../solvers/Electrostatic2D3V/ElectrostaticTwoStream2D3V.hpp"

#include <cmath>
#include <filesystem>
#include <memory>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using boost::math::statistics::simple_ordinary_least_squares;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

#ifndef FIELD_TYPE
#define FIELD_TYPE ContField
#endif

TEST(Electrostatic2D3V, TwoStream) {
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  string vDriverModule;
  DriverSharedPtr drv;

  int argc = 3;
  char *argv[3];

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path conditions_file =
      source_dir / "two_stream_conditions.xml";
  std::filesystem::path mesh_file = source_dir / "two_stream_mesh.xml";

  copy_to_cstring(std::string("test_two_stream"), &argv[0]);
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);

  // Create driver
  session->LoadSolverInfo("Driver", vDriverModule, "Standard");
  drv = GetDriverFactory().CreateInstance(vDriverModule, session, graph);

  auto electrostatic_two_stream_2d3v =
      std::make_shared<ElectrostaticTwoStream2D3V<FIELD_TYPE>>(session, graph,
                                                               drv);

  // space to store energy
  std::vector<double> potential_energy;
  std::vector<double> total_energy;
  std::vector<double> time_steps;
  const int num_time_steps = electrostatic_two_stream_2d3v->num_time_steps;
  potential_energy.reserve(num_time_steps);
  total_energy.reserve(num_time_steps);
  time_steps.reserve(num_time_steps);

  // call back function that executes the energy computation loops and records
  // the outputs
  std::function<void(ElectrostaticTwoStream2D3V<FIELD_TYPE> *)> collect_energy =
      [&](ElectrostaticTwoStream2D3V<FIELD_TYPE> *state) {
        const int time_step = state->time_step;
        if ((time_step > 340) && (time_step < 1420) && (time_step % 20 == 0)) {
          state->potential_energy->compute();
          state->kinetic_energy->compute();
          const double pe = state->potential_energy->energy;
          const double ke = state->kinetic_energy->energy;
          const double te = 0.5 * pe + ke;
          potential_energy.push_back(std::log(pe));
          total_energy.push_back(te);
          time_steps.push_back(time_step * state->charged_particles->dt);
        }
      };
  electrostatic_two_stream_2d3v->push_callback(collect_energy);

  // run the simulation
  electrostatic_two_stream_2d3v->run();

  auto [c0, c1] = simple_ordinary_least_squares(time_steps, potential_energy);

  // check the energy growth rate against the theory
  EXPECT_NEAR(c1, 7.255197456936871, 0.73);

  // check the energy conservation over the whole simulation
  ASSERT_NEAR(total_energy[0] / std::abs(total_energy[0]),
              total_energy[total_energy.size() - 1] / std::abs(total_energy[0]),
              1.0e-4);

  electrostatic_two_stream_2d3v->finalise();
  session->Finalise();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
