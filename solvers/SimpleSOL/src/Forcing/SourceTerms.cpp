///////////////////////////////////////////////////////////////////////////////
//
// File: SourceTerms.cpp
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
// Description: Forcing for axi-symmetric flow.
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/core/ignore_unused.hpp>

#include "SourceTerms.h"

using namespace std;

namespace Nektar {
std::string SourceTerms::className =
    SolverUtils::GetForcingFactory().RegisterCreatorFunction(
        "SourceTerms", SourceTerms::create, "Source terms for 1D SOL code");

SourceTerms::SourceTerms(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const std::weak_ptr<SolverUtils::EquationSystem> &pEquation)
    : Forcing(pSession, pEquation) {}

void SourceTerms::v_InitObject(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
    const unsigned int &pNumForcingFields, const TiXmlElement *pForce) {
  boost::ignore_unused(pForce);

  int spacedim = pFields[0]->GetGraph()->GetSpaceDimension();
  int nPoints = pFields[0]->GetTotPoints();

  m_NumVariable = pNumForcingFields;
  m_varConv =
      MemoryManager<VariableConverter>::AllocateSharedPtr(m_session, spacedim);
  m_x = Array<OneD, NekDouble>(nPoints);
  m_y = Array<OneD, NekDouble>(nPoints);
  pFields[0]->GetCoords(m_x, m_y);

  //===== Set up source term constants from session =====
  // Source term normalisation is calculated relative to the sigma=2 case
  constexpr NekDouble sigma0 = 2.0;

  // xmax should be determined from max(m_x) for all tasks... just set it via a
  // parameter for now.
  m_session->LoadParameter("srcs_xmax", m_xmax, 110.0);

  // (Gaussian sources always positioned halfway along x dimension)
  m_mu = m_xmax / 2;

  // Set normalisation factors for the chosen sigma
  m_session->LoadParameter("srcs_sigma", m_sigma, 2.0);
  m_rho_prefac = 3.989422804e-22 * 1e21 * sigma0 / m_sigma;
  ;
  m_u_prefac = 7.296657414e-27 * -1e26 * sigma0 / m_sigma;
  m_E_prefac = 7.978845608e-5 * 30000.0 * sigma0 / m_sigma;
}

NekDouble CalcGaussian(NekDouble prefac, NekDouble mu, NekDouble sigma,
                       NekDouble x) {
  return prefac * exp(-(mu - x) * (mu - x) / 2 / sigma / sigma);
}

void SourceTerms::v_Apply(
    const Array<OneD, MultiRegions::ExpListSharedPtr> &pFields,
    const Array<OneD, Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble &time) {
  boost::ignore_unused(time);
  unsigned short ndims = pFields[0]->GetGraph()->GetSpaceDimension();

  unsigned short rho_idx = 0;
  unsigned short u_idx = 1;
  unsigned short E_idx = ndims + 1;

  // S^n source term
  for (int i = 0; i < outarray[0].size(); ++i) {
    outarray[rho_idx][i] += CalcGaussian(m_rho_prefac, m_mu, m_sigma, m_x[i]);
  }
  // S^u source term
  for (int i = 0; i < outarray[1].size(); ++i) {
    outarray[u_idx][i] +=
        (m_x[i] / m_mu - 1.) * CalcGaussian(m_u_prefac, m_mu, m_sigma, m_x[i]);
  }
  // S^E source term - divided by 2 since the LHS of the energy equation has
  // been doubled (see README for details)
  for (int i = 0; i < outarray[2].size(); ++i) {
    outarray[E_idx][i] += CalcGaussian(m_E_prefac, m_mu, m_sigma, m_x[i]) / 2.0;
  }
}

} // namespace Nektar
