#include <gtest/gtest.h>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>

#include "../../../../solvers/Electrostatic2D3V/ElectrostaticTwoStream2D3V.hpp"

#include <filesystem>
#include <memory>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

#ifndef FIELD_TYPE
#define FIELD_TYPE ContField
#endif

TEST(Electrostatic2D3V, TwoStream) {
/*
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
  electrostatic_two_stream_2d3v->run();

  electrostatic_two_stream_2d3v->finalise();
  session->Finalise();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
  */
}
