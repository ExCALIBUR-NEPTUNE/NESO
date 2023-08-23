///////////////////////////////////////////////////////////////////////////////
//
// File HWSystem.cpp
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

#include "HWSystem.h"

namespace Nektar {
std::string HWSystem::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "HWLAPD", HWSystem::create,
        "(2D) Hasegawa-Waketani equation system as an intermediate step "
        "towards the full H3-LAPD problem");

HWSystem::HWSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
                   const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), AdvectionSystem(pSession, pGraph),
      H3LAPDSystem(pSession, pGraph) {
  m_required_flds = {"ne", "Gd", "Ge", "w", "phi"};
}

void HWSystem::ExplicitTimeInt(
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
  int Ge_idx = m_field_to_index.get_idx("Ge");
  int Gd_idx = m_field_to_index.get_idx("Gd");
  int phi_idx = m_field_to_index.get_idx("phi");
  int w_idx = m_field_to_index.get_idx("w");

  // Advect both ne and w using ExB velocity
  AddAdvTerms({"ne"}, m_advElec, m_vExB, inarray, outarray, time);
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
  Vmath::Vsub(nPts, outarray[ne_idx], 1, HWterm_2D_kappa, 1, outarray[ne_idx],
              1);
}

// Set Phi solve RHS = w
void HWSystem::GetPhiSolveRHS(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, NekDouble> &rhs) {
  int nPts = GetNpoints();
  int w_idx = m_field_to_index.get_idx("w");
  // OP: Orig version has RHS=w, presumably it ought be alpha*w? :
  // Vmath::Smul(nPts, m_alpha, inarray[w_idx], 1, rhs, 1);
  Vmath::Vcopy(nPts, inarray[w_idx], 1, rhs, 1);
}

void HWSystem::LoadParams() {
  H3LAPDSystem::LoadParams();

  // alpha
  m_session->LoadParameter("alpha", m_alpha, 2);

  // kappa
  m_session->LoadParameter("kappa", m_kappa, 1);
}

} // namespace Nektar
