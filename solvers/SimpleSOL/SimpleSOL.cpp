///////////////////////////////////////////////////////////////////////////////
//
// File: SimpleSOL.cpp
//
//
// Description: Driver for SimpleSOL.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>

#include "SimpleSOL.h"

using namespace Nektar;
using namespace Nektar::SolverUtils;

namespace NESO {
namespace Solvers {
int run_SimpleSOL(int argc, char *argv[]) {
  try {
    // Create session reader.
    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);

    // Read the mesh and create a MeshGraph object.
    auto graph = SpatialDomains::MeshGraph::Read(session);

    // Create driver.
    std::string driverName;
    session->LoadSolverInfo("Driver", driverName, "Standard");
    auto drv = GetDriverFactory().CreateInstance(driverName, session, graph);

    // Execute driver
    drv->Execute();

    // Finalise session
    session->Finalise();
  } catch (const std::runtime_error &e) {
    return 1;
  } catch (const std::string &eStr) {
    std::cout << "Error: " << eStr << std::endl;
  }

  return 0;
}

} // namespace Solvers
} // namespace NESO