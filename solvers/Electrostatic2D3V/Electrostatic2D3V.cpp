///////////////////////////////////////////////////////////////////////////////
//
// File CompressibleFlowSolver.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
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
// Description: Compressible Flow Equations framework solver
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>

#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>

#include "Diagnostics/field_energy.hpp"
#include "ParticleSystems/charged_particles.hpp"
#include "ParticleSystems/poisson_particle_coupling.hpp"

#include <memory>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;

#define FIELD_TYPE ContField

int main(int argc, char *argv[]) {
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  string vDriverModule;
  DriverSharedPtr drv;

  try {
    // Create session reader.
    session = LibUtilities::SessionReader::CreateInstance(argc, argv);

    // Create MeshGraph.
    graph = SpatialDomains::MeshGraph::Read(session);

    // Create driver
    session->LoadSolverInfo("Driver", vDriverModule, "Standard");
    drv = GetDriverFactory().CreateInstance(vDriverModule, session, graph);

    auto charged_particles = std::make_shared<ChargedParticles>(session, graph);
    auto poisson_particle_coupling =
        std::make_shared<PoissonParticleCoupling<FIELD_TYPE>>(
            session, graph, drv, charged_particles);

    int num_time_steps;
    session->LoadParameter("particle_num_time_steps", num_time_steps);
    int num_write_particle_steps;
    session->LoadParameter("particle_num_write_particle_steps",
                           num_write_particle_steps);
    int num_write_field_steps;
    session->LoadParameter("particle_num_write_field_steps",
                           num_write_field_steps);
    int num_write_field_energy_steps;
    session->LoadParameter("particle_num_write_field_energy_steps",
                           num_write_field_energy_steps);
    int num_print_steps;
    session->LoadParameter("particle_num_print_steps", num_print_steps);

    auto field_energy = std::make_shared<FieldEnergy<FIELD_TYPE>>(
        poisson_particle_coupling->potential_function, "field_energy.h5");

    for (int stepx = 0; stepx < num_time_steps; stepx++) {
      auto t0 = profile_timestamp();

      charged_particles->velocity_verlet_1();

      poisson_particle_coupling->compute_field();

      charged_particles->velocity_verlet_2();

      // writes trajectory
      if (num_write_particle_steps > 0) {
        if ((stepx % num_write_particle_steps) == 0) {
          charged_particles->write();
        }
      }
      if (num_write_field_steps > 0) {
        if ((stepx % num_write_field_steps) == 0) {
          poisson_particle_coupling->write_forcing(stepx);
          poisson_particle_coupling->write_potential(stepx);
        }
      }
      if (num_write_field_energy_steps > 0) {
        if ((stepx % num_write_field_energy_steps) == 0) {
          field_energy->write();
        }
      }
      if (num_print_steps > 0) {
        if ((stepx % num_print_steps) == 0) {
          if (charged_particles->sycl_target->comm_pair.rank_parent == 0) {
            nprint("step:", stepx, profile_elapsed(t0, profile_timestamp()),
                   "field energy:", field_energy->l2_energy);
          }
        }
      }
    }

    // Print out timings if verbose
    if (session->DefinesCmdLineArgument("verbose")) {
      int iolevel;

      session->LoadParameter("IO_Timer_Level", iolevel, 1);

      LibUtilities::Timer::PrintElapsedRegions(session->GetComm(), std::cout,
                                               iolevel);
    }

    field_energy->close();
    charged_particles->free();

    // Finalise communications
    session->Finalise();
  } catch (const std::runtime_error &) {
    return 1;
  } catch (const std::string &eStr) {
    cout << "Error: " << eStr << endl;
  }

  return 0;
}
