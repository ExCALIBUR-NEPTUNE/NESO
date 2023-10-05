///////////////////////////////////////////////////////////////////////////////
//
// Description: Entrypoint for the evaluation benchmark.
//
///////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <mpi.h>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <string>
#include <map>


#include <nektar_interface/particle_interface.hpp>
#include <nektar_interface/equation_system_wrapper.hpp>
using namespace NESO;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

int main_evaluation(int argc, char *argv[], LibUtilities::SessionReaderSharedPtr session);
