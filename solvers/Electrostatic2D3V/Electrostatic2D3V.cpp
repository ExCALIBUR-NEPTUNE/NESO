#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>
#include <SpatialDomains/MeshGraphIO.h>

#include "ElectrostaticTwoStream2D3V.hpp"

#include <memory>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;

#ifndef FIELD_TYPE
#define FIELD_TYPE ContField
#endif

int main(int argc, char *argv[]) {
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  string vDriverModule;
  DriverSharedPtr drv;

  try {
    // Create session reader.
    session = LibUtilities::SessionReader::CreateInstance(argc, argv);

    // Create MeshGraph.
    graph = SpatialDomains::MeshGraphIO::Read(session);

    // Create driver
    session->LoadSolverInfo("Driver", vDriverModule, "Standard");
    drv = GetDriverFactory().CreateInstance(vDriverModule, session, graph);

    auto electrostatic_two_stream_2d3v =
        std::make_shared<ElectrostaticTwoStream2D3V<FIELD_TYPE>>(session, graph,
                                                                 drv);
    electrostatic_two_stream_2d3v->run();

    // Print out timings if verbose
    if (session->DefinesCmdLineArgument("verbose")) {
      int iolevel;

      session->LoadParameter("IO_Timer_Level", iolevel, 1);

      LibUtilities::Timer::PrintElapsedRegions(session->GetComm(), std::cout,
                                               iolevel);
    }

    electrostatic_two_stream_2d3v->finalise();
    // Finalise communications
    session->Finalise();
  } catch (const std::runtime_error &) {
    return 1;
  } catch (const std::string &eStr) {
    cout << "Error: " << eStr << endl;
  }

  return 0;
}
