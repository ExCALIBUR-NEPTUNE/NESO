#include <iostream>
#include <memory>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>
#include <SpatialDomains/MeshGraphIO.h>

using namespace Nektar;
using namespace Nektar::SolverUtils;

#ifndef FIELD_TYPE
#define FIELD_TYPE ContField
#endif

void mesh_plotting_inner(int argc, char **argv,
                         LibUtilities::SessionReaderSharedPtr session,
                         SpatialDomains::MeshGraphSharedPtr graph);

void mesh_plotting_inner_halos(int argc, char **argv,
                               LibUtilities::SessionReaderSharedPtr session,
                               SpatialDomains::MeshGraphSharedPtr graph,
                               const int halo_stencil_width,
                               const int halo_stencil_pbc);

int main(int argc, char *argv[]) {
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  std::string vDriverModule;
  // DriverSharedPtr drv;

  try {
    // Create session reader.
    session = LibUtilities::SessionReader::CreateInstance(argc, argv);

    // Create MeshGraph.
    graph = SpatialDomains::MeshGraphIO::Read(session);

    // Create driver
    // session->LoadSolverInfo("Driver", vDriverModule, "Standard");
    // drv = GetDriverFactory().CreateInstance(vDriverModule, session, graph);

    // Print out timings if verbose
    if (session->DefinesCmdLineArgument("verbose")) {
      int iolevel;

      session->LoadParameter("IO_Timer_Level", iolevel, 1);

      LibUtilities::Timer::PrintElapsedRegions(session->GetComm(), std::cout,
                                               iolevel);
    }

    int halo_stencil_width = 0;
    int halo_stencil_pbc = 1;
    session->LoadParameter("halo_stencil_width", halo_stencil_width, 0);
    session->LoadParameter("halo_stencil_pbc", halo_stencil_pbc, 1);
    mesh_plotting_inner(argc, argv, session, graph);

    for (int sx = 0; sx < (halo_stencil_width + 1); sx++) {
      mesh_plotting_inner_halos(argc, argv, session, graph, sx,
                                halo_stencil_pbc);
    }

    // Finalise communications
    session->Finalise();
  } catch (const std::runtime_error &) {
    return 1;
  } catch (const std::string &eStr) {
    std::cout << "Error: " << eStr << std::endl;
  }

  return 0;
}
