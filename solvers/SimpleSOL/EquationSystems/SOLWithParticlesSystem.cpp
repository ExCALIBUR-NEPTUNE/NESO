///////////////////////////////////////////////////////////////////////////////
//
// File SOLWithParticlesSystem.cpp
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
// Description: Adds particles to SOLSystem
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/algorithm/string/predicate.hpp>

#include "SOLWithParticlesSystem.h"

namespace Nektar {
string SOLWithParticlesSystem::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "SOLWithParticles", SOLWithParticlesSystem::create,
        "SOL equations with particle source terms.");

SOLWithParticlesSystem::SOLWithParticlesSystem(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), AdvectionSystem(pSession, pGraph),
      SOLSystem(pSession, pGraph), m_field_to_index(pSession->GetVariables()) {

  m_particle_sys = std::make_shared<NeutralParticleSystem>(pSession, pGraph);
  m_required_flds.push_back("E_src");
  m_required_flds.push_back("rho_src");
  m_required_flds.push_back("rhou_src");
  m_required_flds.push_back("rhov_src");
  m_required_flds.push_back("T");

  // mass recording diagnostic creation
  m_diag_mass_recording_enabled =
      pSession->DefinesParameter("mass_recording_step");
}

void SOLWithParticlesSystem::UpdateTemperature() {
  // Compute initial T vals
  // N.B. GetTemperature requires field order rho,rhou,[rhov],[rhow],E
  int nFields_for_Tcalc = m_field_to_index.get_idx("E") + 1;
  Array<OneD, Array<OneD, NekDouble>> physvals(nFields_for_Tcalc);
  for (int i = 0; i < nFields_for_Tcalc; ++i) {
    physvals[i] = m_fields[i]->GetPhys();
  }
  auto Tfield = m_fields[m_field_to_index.get_idx("T")];
  m_varConv->GetTemperature(physvals, Tfield->UpdatePhys());
  Tfield->FwdTrans(Tfield->GetPhys(),
                   Tfield->UpdateCoeffs()); // May not be needed
}

void SOLWithParticlesSystem::v_InitObject(bool DeclareField) {
  SOLSystem::v_InitObject(DeclareField);

  // Set particle timestep from params
  m_session->LoadParameter("num_particle_steps_per_fluid_step",
                           m_num_part_substeps, 1);
  m_session->LoadParameter("particle_num_write_particle_steps",
                           m_num_write_particle_steps, 0);
  m_part_timestep = m_timestep / m_num_part_substeps;

  // Store DisContFieldSharedPtr casts of fields in a map, indexed by name, for
  // use in particle project,evaluate operations
  int idx = 0;
  for (auto &field_name : m_session->GetVariables()) {
    m_discont_fields[field_name] =
        std::dynamic_pointer_cast<MultiRegions::DisContField>(m_fields[idx]);
    idx++;
  }

  m_particle_sys->setup_project(
      m_discont_fields["rho_src"], m_discont_fields["rhou_src"],
      m_discont_fields["rhov_src"], m_discont_fields["E_src"]);

  m_particle_sys->setup_evaluate_n(m_discont_fields["rho"]);
  m_particle_sys->setup_evaluate_T(m_discont_fields["T"]);

  m_diag_mass_recording =
      std::make_shared<MassRecording<MultiRegions::DisContField>>(
          m_session, m_particle_sys, m_discont_fields["rho"]);
}

/**
 * @brief Destructor for SOLWithParticlesSystem class.
 */
SOLWithParticlesSystem::~SOLWithParticlesSystem() { m_particle_sys->free(); }

bool SOLWithParticlesSystem::v_PostIntegrate(int step) {
  // Writes a step of the particle trajectory.
  if (m_num_write_particle_steps > 0) {
    if ((step % m_num_write_particle_steps) == 0) {
      m_particle_sys->write(step);
      m_particle_sys->write_source_fields();
    }
  }

  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recording->compute(step);
  }

  m_solver_callback_handler.call_post_integrate(this);
  return SOLSystem::v_PostIntegrate(step);
}

bool SOLWithParticlesSystem::v_PreIntegrate(int step) {
  m_solver_callback_handler.call_pre_integrate(this);

  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recording->compute_initial_fluid_mass();
  }
  //  Update Temperature field
  UpdateTemperature();
  // Integrate the particle system to the requested time.
  m_particle_sys->integrate(m_time + m_timestep, m_part_timestep);
  // Project onto the source fields
  m_particle_sys->project_source_terms();

  return SOLSystem::v_PreIntegrate(step);
}

ExpListSharedPtr
SOLWithParticlesSystem::GetField(const std::string field_name) {
  ExpListSharedPtr ptr(nullptr);
  int idx = m_field_to_index.get_idx(field_name);
  if (idx > -1) {
    ptr = m_fields[idx];
  }
  return ptr;
}

std::shared_ptr<NeutralParticleSystem>
SOLWithParticlesSystem::GetNeutralParticleSystem() {
  return m_particle_sys;
}

} // namespace Nektar
