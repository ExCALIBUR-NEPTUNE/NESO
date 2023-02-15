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
      SOLSystem(pSession, pGraph) {

  m_particle_sys = std::make_shared<NeutralParticleSystem>(pSession, pGraph);
}

void SOLWithParticlesSystem::v_InitObject(bool DeclareField) {
  SOLSystem::v_InitObject(DeclareField);

  // Set particle timestep from params
  m_session->LoadParameter("num_particle_steps_per_fluid_step",
                           m_num_part_substeps, 1);
  m_session->LoadParameter("particle_num_write_particle_steps",
                           m_num_write_particle_steps, 0);
  m_part_timestep = m_timestep / m_num_part_substeps;

  // Store DisContFieldSharedPtr casts of *_src fields in a map, indexed by
  // name, to simplify projection of particle quantities later
  int idx = 0;
  for (auto &field_name : m_session->GetVariables()) {
    m_src_fields[field_name] =
        std::dynamic_pointer_cast<MultiRegions::DisContField>(m_fields[idx]);
    idx++;
  }

  m_particle_sys->setup_project(m_src_fields["rho_src"]);
  m_particle_sys->setup_evaluate_rho(m_src_fields["rho"]);
  m_particle_sys->setup_evaluate_E(m_src_fields["E"]);

  // Use an augmented version of SOLSystem's DefineOdeRhs()
  m_ode.DefineOdeRhs(&SOLWithParticlesSystem::DoOdeRhs, this);
}

/**
 * @brief Destructor for SOLWithParticlesSystem class.
 */
SOLWithParticlesSystem::~SOLWithParticlesSystem() { m_particle_sys->free(); }

/**
 * @brief Compute the right-hand side.
 */
void SOLWithParticlesSystem::DoOdeRhs(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {

  // Integrate the particle system to the requested time.
  m_particle_sys->integrate(time, m_part_timestep);
  // Project onto the source fields
  m_particle_sys->project_source_terms();

  // Energy source field already calculated by the particle system? If not, do
  // so here

  // Parent implementation adds relevant source term fields to the RHS of each
  // eqn; see SourceTerms::v_Apply()
  SOLSystem::DoOdeRhs(inarray, outarray, time);

  // // Zero the source term fields here?
  // for (auto name_field_pair : this->m_src_fields) {
  //   Vmath::Zero(name_field_pair.second->GetTotPoints(),
  //               name_field_pair.second->UpdatePhys(), 1);
  // }
}

bool SOLWithParticlesSystem::v_PostIntegrate(int step) {
  // Writes a step of the particle trajectory.
  if (m_num_write_particle_steps > 0) {
    if ((step % m_num_write_particle_steps) == 0) {
      m_particle_sys->write(step);
    }
  }
  return SOLSystem::v_PostIntegrate(step);
}

} // namespace Nektar
