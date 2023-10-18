///////////////////////////////////////////////////////////////////////////////
//
// File 2Din3DHW.cpp
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
// Description: 2D Hasegawa-Waketani equation system as an intermediate step
// towards the full H3-LAPD problem.  Implemented by Ed Threlfall in August 2023
// after realizing he didn't know how to do numerical flux terms in 3D
// Hasegawa-Wakatani. Parameter choices are same as in Nektar-Driftwave 2D
// proxyapp. Evolves ne, w, phi only, no momenta, no ions
//
///////////////////////////////////////////////////////////////////////////////
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

#include "HW2Din3DSystem.hpp"

namespace Nektar {
std::string HW2Din3DSystem::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "2Din3DHW", HW2Din3DSystem::create,
        "(2D) Hasegawa-Waketani equation system as an intermediate step "
        "towards the full H3-LAPD problem");

HW2Din3DSystem::HW2Din3DSystem(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), AdvectionSystem(pSession, pGraph),
      DriftReducedSystem(pSession, pGraph) {
  m_required_flds = {"ne", "w", "phi"};
  m_int_fld_names = {"ne", "w"};

  // Frequency of growth rate recording. Set zero to disable.
  m_diag_growth_rates_recording_enabled =
      pSession->DefinesParameter("growth_rates_recording_step");
}
void HW2Din3DSystem::CalcEAndAdvVels(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray) {
  DriftReducedSystem::CalcEAndAdvVels(inarray);
  int nPts = GetNpoints();

  // int ne_idx = m_field_to_index.get_idx("ne");
  // int Gd_idx = m_field_to_index.get_idx("Gd");
  // int Ge_idx = m_field_to_index.get_idx("Ge");

  Vmath::Zero(nPts, m_vParElec, 1);
  // vAdv[iDim] = b[iDim]*v_par + v_ExB[iDim] for each species
  for (auto iDim = 0; iDim < m_graph->GetSpaceDimension(); iDim++) {
    Vmath::Svtvp(nPts, m_b_unit[iDim], m_vParElec, 1, m_vExB[iDim], 1,
                 m_vAdvElec[iDim], 1);
  }
}

void HW2Din3DSystem::ExplicitTimeInt(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {

  // Check inarray for NaNs
  for (auto &var : {"ne", "w"}) {
    auto fidx = m_field_to_index.get_idx(var);
    for (auto ii = 0; ii < inarray[fidx].size(); ii++) {
      if (!std::isfinite(inarray[fidx][ii])) {
        std::cout << "NaN in field " << var << ", aborting." << std::endl;
        exit(1);
      }
    }
  }

  ZeroOutArray(outarray);

  // Solver for electrostatic potential.
  SolvePhi(inarray);

  // Calculate electric field from Phi, as well as corresponding drift velocity
  CalcEAndAdvVels(inarray);

  // Get field indices
  int nPts = GetNpoints();
  int ne_idx = m_field_to_index.get_idx("ne");
  int phi_idx = m_field_to_index.get_idx("phi");
  int w_idx = m_field_to_index.get_idx("w");

  // Advect ne and w (m_vAdvElec === m_vExB for HW)
  AddAdvTerms({"ne"}, m_advElec, m_vAdvElec, inarray, outarray, time);
  AddAdvTerms({"w"}, m_advVort, m_vExB, inarray, outarray, time);

  // Add \alpha*(\phi-n_e) to RHS
  Array<OneD, NekDouble> HWterm_2D_alpha(nPts);
  Vmath::Vsub(nPts, m_fields[phi_idx]->GetPhys(), 1,
              m_fields[ne_idx]->GetPhys(), 1, HWterm_2D_alpha, 1);
  Vmath::Smul(nPts, m_alpha, HWterm_2D_alpha, 1, HWterm_2D_alpha, 1);
  Vmath::Vadd(nPts, outarray[w_idx], 1, HWterm_2D_alpha, 1, outarray[w_idx], 1);
  Vmath::Vadd(nPts, outarray[ne_idx], 1, HWterm_2D_alpha, 1, outarray[ne_idx],
              1);

  // Add \kappa*\dpartial\phi/\dpartial y to RHS
  Array<OneD, NekDouble> HWterm_2D_kappa(nPts);
  m_fields[phi_idx]->PhysDeriv(1, m_fields[phi_idx]->GetPhys(),
                               HWterm_2D_kappa);
  Vmath::Smul(nPts, m_kappa, HWterm_2D_kappa, 1, HWterm_2D_kappa, 1);
  Vmath::Vsub(nPts, outarray[ne_idx], 1, HWterm_2D_kappa, 1, outarray[ne_idx],
              1);

  // Add particle sources
  AddParticleSources({"ne"}, outarray);
}

// Set Phi solve RHS = w
void HW2Din3DSystem::GetPhiSolveRHS(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, NekDouble> &rhs) {
  int nPts = GetNpoints();
  int w_idx = m_field_to_index.get_idx("w");
  // OP: Orig version has RHS=w, presumably it ought be alpha*w? :
  // Vmath::Smul(nPts, m_alpha, inarray[w_idx], 1, rhs, 1);
  Vmath::Vcopy(nPts, inarray[w_idx], 1, rhs, 1);
}

void HW2Din3DSystem::LoadParams() {
  DriftReducedSystem::LoadParams();

  // alpha
  m_session->LoadParameter("HW_alpha", m_alpha, 2);

  // kappa
  m_session->LoadParameter("HW_kappa", m_kappa, 1);
}

/**
 * @brief Initialization for HW2Din3DSystem class.
 */
void HW2Din3DSystem::v_InitObject(bool DeclareField) {
  DriftReducedSystem::v_InitObject(DeclareField);

  // Bind RHS function for time integration object
  m_ode.DefineOdeRhs(&HW2Din3DSystem::ExplicitTimeInt, this);

  // Create diagnostic for recording growth rates
  m_diag_growth_rates_recorder =
      std::make_shared<GrowthRatesRecorder<MultiRegions::DisContField>>(
          m_session, m_particle_sys, m_discont_fields["ne"],
          m_discont_fields["w"], m_discont_fields["phi"], GetNpoints(), m_alpha,
          m_kappa);
}

bool HW2Din3DSystem::v_PostIntegrate(int step) {
  if (m_diag_growth_rates_recording_enabled) {
    m_diag_growth_rates_recorder->compute(step);
  }
  m_solver_callback_handler.call_post_integrate(this);
  return DriftReducedSystem::v_PostIntegrate(step);
}

} // namespace Nektar
