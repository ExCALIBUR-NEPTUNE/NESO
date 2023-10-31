///////////////////////////////////////////////////////////////////////////////
//
// File: H3LAPD.cpp
//
//
// Description: Solver function for H3LAPD.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>

#include "H3LAPD.hpp"

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::H3LAPD {

int run_H3LAPD(int argc, char *argv[]) {
  try {
    // Create session reader.
    auto session = LU::SessionReader::CreateInstance(argc, argv);

    // Read the mesh and create a MeshGraph object.
    auto graph = SD::MeshGraph::Read(session);

    // Create driver.
    std::string driverName;
    session->LoadSolverInfo("Driver", driverName, "Standard");
    auto drv =
        SU::GetDriverFactory().CreateInstance(driverName, session, graph);

    // Execute driver
    drv->Execute();

    // Finalise session
    session->Finalise();
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  } catch (const std::string &eStr) {
    std::cerr << "Error: " << eStr << std::endl;
    return 2;
  }

  return 0;
}

} // namespace NESO::Solvers::H3LAPD
