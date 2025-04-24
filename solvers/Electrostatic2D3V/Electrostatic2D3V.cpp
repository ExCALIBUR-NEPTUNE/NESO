#include <memory>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <SolverUtils/Driver.h>
#include <SpatialDomains/MeshGraphIO.h>

#include "ElectrostaticTwoStream2D3V.hpp"

namespace ES2D3V = NESO::Solvers::Electrostatic2D3V;
namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

#ifndef FIELD_TYPE
#define FIELD_TYPE ContField
#endif

int main(int argc, char *argv[]) {
  LU::SessionReaderSharedPtr session;
  SD::MeshGraphSharedPtr graph;
  std::string driver_module;
  SU::DriverSharedPtr drv;

  try {
    // Create session reader.
    session = LU::SessionReader::CreateInstance(argc, argv);

    // Create MeshGraph.
    graph = SD::MeshGraphIO::Read(session);

    // Create driver
    session->LoadSolverInfo("Driver", driver_module, "Standard");
    drv = SU::GetDriverFactory().CreateInstance(driver_module, session, graph);

    auto electrostatic_two_stream_2d3v =
        std::make_shared<ES2D3V::ElectrostaticTwoStream2D3V<FIELD_TYPE>>(
            session, graph, drv);
    electrostatic_two_stream_2d3v->run();

    // Print out timings if verbose
    if (session->DefinesCmdLineArgument("verbose")) {
      int iolevel;

      session->LoadParameter("IO_Timer_Level", iolevel, 1);

      LU::Timer::PrintElapsedRegions(session->GetComm(), std::cout, iolevel);
    }

    electrostatic_two_stream_2d3v->finalise();
    // Finalise communications
    session->Finalise();
  } catch (const std::runtime_error &) {
    return 1;
  } catch (const std::string &eStr) {
    std::cout << "Error: " << eStr << std::endl;
  }

  return 0;
}
