#include <iostream>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <SpatialDomains/MeshGraphIO.h>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;

#ifndef FIELD_TYPE
#define FIELD_TYPE ContField
#endif

namespace NESO::MeshPlotting {

void mesh_plotting_inner(int argc, char **argv,
                         LU::SessionReaderSharedPtr session,
                         SD::MeshGraphSharedPtr graph);

void mesh_plotting_inner_halos(int argc, char **argv,
                               LU::SessionReaderSharedPtr session,
                               SD::MeshGraphSharedPtr graph,
                               const int halo_stencil_width,
                               const int halo_stencil_pbc);
} // namespace NESO::MeshPlotting

int main(int argc, char *argv[]) {
  LU::SessionReaderSharedPtr session;
  SD::MeshGraphSharedPtr graph;
  std::string vDriverModule;

  try {
    // Create session reader.
    session = LU::SessionReader::CreateInstance(argc, argv);

    // Create MeshGraph.
    graph = SD::MeshGraphIO::Read(session);

    // Print out timings if verbose
    if (session->DefinesCmdLineArgument("verbose")) {
      int iolevel;

      session->LoadParameter("IO_Timer_Level", iolevel, 1);

      LU::Timer::PrintElapsedRegions(session->GetComm(), std::cout, iolevel);
    }

    int halo_stencil_width = 0;
    int halo_stencil_pbc = 1;
    session->LoadParameter("halo_stencil_width", halo_stencil_width, 0);
    session->LoadParameter("halo_stencil_pbc", halo_stencil_pbc, 1);
    NESO::MeshPlotting::mesh_plotting_inner(argc, argv, session, graph);

    for (int sx = 0; sx < (halo_stencil_width + 1); sx++) {
      NESO::MeshPlotting::mesh_plotting_inner_halos(argc, argv, session, graph,
                                                    sx, halo_stencil_pbc);
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
