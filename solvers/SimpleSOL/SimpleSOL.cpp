///////////////////////////////////////////////////////////////////////////////
//
// File: SOLSolver.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Driver for the SOL solver.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>

using namespace Nektar;
using namespace Nektar::SolverUtils;

int main(int argc, char *argv[]) {
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
