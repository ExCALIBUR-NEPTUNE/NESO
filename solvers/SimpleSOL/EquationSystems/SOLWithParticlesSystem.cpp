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

#include "SOLWithParticlesSystem.h"

namespace Nektar {
string SOLWithParticlesSystem::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "SOLWithParticles", SOLWithParticlesSystem::create,
        "SOL equations with particle source terms.");

SOLWithParticlesSystem::SOLWithParticlesSystem(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : m_particle_sys(pSession, pGraph), UnsteadySystem(pSession, pGraph),
      AdvectionSystem(pSession, pGraph), SOLSystem(pSession, pGraph) {}

void SOLWithParticlesSystem::v_InitObject(bool DeclareField) {
  SOLSystem::v_InitObject(DeclareField);

  m_session->LoadParameter("num_particle_steps_per_fluid_step",
                           m_num_part_substeps, 1);
  m_session->LoadParameter("particle_num_write_particle_steps",
                           m_num_write_particle_steps, 0);

  // Any additional particle init needed after construction?
  m_part_timestep = m_timestep / m_num_part_substeps;

  // Use an augmented version of SOLSystem's DefineOdeRhs()
  m_ode.DefineOdeRhs(&SOLWithParticlesSystem::DoOdeRhs, this);
}

/**
 * @brief Destructor for SOLWithParticlesSystem class.
 */
SOLWithParticlesSystem::~SOLWithParticlesSystem() { m_particle_sys.free(); }


/**
 * @brief Compute the right-hand side.
 */
void SOLWithParticlesSystem::DoOdeRhs(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {

  // Integrate the particle system to the requested time.
  m_particle_sys.integrate(time, m_part_timestep);
  // TODO project onto the fields

  // Neutrals have been ionised, so no project onto the  delta_rho field
  // ...
  // Now do rho = rho + delta_rho
  //const int rho_index = this->GetFieldIndex("rho");
  //const int delta_rho_index = this->GetFieldIndex("delta_rho");
  //Vmath::Vadd(m_fields[rho_index]->GetTotPoints(),
  //            m_fields[rho_index]->UpdateCoeffs(), 1
  //            m_fields[delta_rho_index]->UpdatePhys(), 1,
  //            m_fields[rho_index]->UpdateCoeffs(), 1); // TODO is this right?
  //now zero the delta_rho field
  //  Vmath::Zero(m_fields[delta_rho_index]->GetTotPoints(),
  //              m_fields[delta_rho_index]->UpdatePhys(), 1);
  // TODO momentum and energy sources
  // Now density source is done do the rest

  SOLSystem::DoOdeRhs(inarray, outarray, time);
}

bool SOLWithParticlesSystem::v_PostIntegrate(int step){
  // Writes a step of the particle trajectory.
  if (m_num_write_particle_steps > 0){
    if((step % m_num_write_particle_steps) == 0){
      m_particle_sys.write(step);
    }
  }
  return SOLSystem::v_PostIntegrate(step);
}



} // namespace Nektar
