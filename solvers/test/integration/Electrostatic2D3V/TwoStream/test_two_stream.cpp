#include <gtest/gtest.h>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <boost/math/statistics/linear_regression.hpp>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>
#include <SpatialDomains/MeshGraphIO.h>

#include "../../../../../solvers/Electrostatic2D3V/ElectrostaticTwoStream2D3V.hpp"

#include <cmath>
#include <filesystem>
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
  graph = SpatialDomains::MeshGraphIO::Read(session);

  // Create driver
  session->LoadSolverInfo("Driver", vDriverModule, "Standard");
  drv = GetDriverFactory().CreateInstance(vDriverModule, session, graph);

  auto electrostatic_two_stream_2d3v =
      std::make_shared<ES2D3V::ElectrostaticTwoStream2D3V<FIELD_TYPE>>(
          session, graph, drv);

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
  std::function<void(ES2D3V::ElectrostaticTwoStream2D3V<FIELD_TYPE> *)>
      collect_energy =
          [&](ES2D3V::ElectrostaticTwoStream2D3V<FIELD_TYPE> *state) {
            const int time_step = state->time_step;
            if ((time_step > 800) && (time_step < 1800) &&
                (time_step % 20 == 0)) {
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
  electrostatic_two_stream_2d3v->push_callback(collect_energy);

  // run the simulation
  electrostatic_two_stream_2d3v->run();

  const int num_energy_steps = potential_energy.size();
  std::vector<double> potential_energy_window;
  std::vector<double> time_steps_window;

  time_steps_window.reserve(num_energy_steps);
  potential_energy_window.reserve(num_energy_steps);

  int index_start = -1;
  int index_end = -1;
  for (int stepx = 1; stepx < (num_energy_steps - 1); stepx++) {
    const double tmp_d2fdx2 =
        std::abs(potential_energy[stepx + 1] - 2.0 * potential_energy[stepx] +
                 potential_energy[stepx - 1]);

    if (tmp_d2fdx2 < 5.0e-3) {
      if (index_start == -1) {
        index_start = stepx;
      }
      index_end = stepx;
    }
  };
  for (int ix = index_start; ix < index_end; ix++) {
    time_steps_window.push_back(time_steps[ix]);
    potential_energy_window.push_back(potential_energy[ix]);
  }

  // Check the energy growth rate against the theory.
  // The theory is described in NEPTUNE report M4c "1-D and 2-D particle
  // models".
  // try the extracted region if possible
  if ((index_start > -1) && (index_start < index_end)) {
    auto [c0, c1] = simple_ordinary_least_squares(time_steps_window,
                                                  potential_energy_window);
    EXPECT_NEAR(c1, 7.255197456936871, 0.73);
  } else {
    auto [c0, c1] = simple_ordinary_least_squares(time_steps, potential_energy);
    EXPECT_NEAR(c1, 7.255197456936871, 0.73);
  }

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
