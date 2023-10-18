///////////////////////////////////////////////////////////////////////////////
//
// File LAPDSystem.cpp
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
// Description: LAPD equation system.
//
///////////////////////////////////////////////////////////////////////////////
#include "LAPDSystem.hpp"
#include <LibUtilities/BasicUtils/Vmath.hpp>

namespace Nektar {
std::string LAPDSystem::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "LAPD", LAPDSystem::create, "LAPD equation system");

LAPDSystem::LAPDSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
                       const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), AdvectionSystem(pSession, pGraph),
      DriftReducedSystem(pSession, pGraph),
      m_vAdvDiffPar(pGraph->GetSpaceDimension()),
      m_vAdvIons(pGraph->GetSpaceDimension()) {
  m_required_flds = {"ne", "Ge", "Gd", "w", "phi"};
  m_int_fld_names = {"ne", "Ge", "Gd", "w"};
  // Construct particle system
  m_particle_sys = std::make_shared<NeutralParticleSystem>(pSession, pGraph);
}

void LAPDSystem::AddCollisionTerms(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray) {

  int npts = inarray[0].size();

  // Field indices
  int ne_idx = m_field_to_index.get_idx("ne");
  int Ge_idx = m_field_to_index.get_idx("Ge");
  int Gd_idx = m_field_to_index.get_idx("Gd");
  int w_idx = m_field_to_index.get_idx("w");

  /*
  Calculate collision term
  This is the momentum(-density) tranferred from electrons to ions by
  collisions, so add it to Gd rhs, but subtract it from Ge rhs
  */
  Array<OneD, NekDouble> collisionFreqs(npts), collisionTerm(npts),
      vDiffne(npts);
  Vmath::Vmul(npts, inarray[ne_idx], 1, m_vAdvDiffPar[2], 1, vDiffne, 1);
  CalcCollisionFreqs(inarray[ne_idx], collisionFreqs);
  for (auto ii = 0; ii < npts; ii++) {
    collisionTerm[ii] = m_me * collisionFreqs[ii] * vDiffne[ii];
  }

  // Subtract collision term from Ge rhs
  Vmath::Vsub(npts, outarray[Ge_idx], 1, collisionTerm, 1, outarray[Ge_idx], 1);

  // Add collision term to Gd rhs
  Vmath::Vadd(npts, outarray[Gd_idx], 1, collisionTerm, 1, outarray[Gd_idx], 1);
}

void LAPDSystem::AddEParTerms(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray) {

  int nPts = GetNpoints();

  // Field indices
  int ne_idx = m_field_to_index.get_idx("ne");
  int Ge_idx = m_field_to_index.get_idx("Ge");
  int Gd_idx = m_field_to_index.get_idx("Gd");

  // Calculate EParTerm = e*n_e*EPar (=== e*n_d*EPar)
  // ***Assumes field aligned with z-axis***
  Array<OneD, NekDouble> EParTerm(nPts);
  Vmath::Vmul(nPts, inarray[ne_idx], 1, m_E[2], 1, EParTerm, 1);
  Vmath::Smul(nPts, m_charge_e, EParTerm, 1, EParTerm, 1);

  // Subtract EParTerm from outarray[Ge_idx]
  Vmath::Vsub(nPts, outarray[Ge_idx], 1, EParTerm, 1, outarray[Ge_idx], 1);

  // Add EParTerm to outarray[Gd_idx]
  Vmath::Vadd(nPts, outarray[Gd_idx], 1, EParTerm, 1, outarray[Gd_idx], 1);
}

void LAPDSystem::AddGradPTerms(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray) {

  int npts = inarray[0].size();

  // Field indices
  int ne_idx = m_field_to_index.get_idx("ne");
  int Ge_idx = m_field_to_index.get_idx("Ge");
  int Gd_idx = m_field_to_index.get_idx("Gd");

  // Subtract parallel pressure gradient for Electrons from outarray[Ge_idx]
  Array<OneD, NekDouble> PElec(npts), parGradPElec(npts);
  Vmath::Smul(npts, m_Te, inarray[ne_idx], 1, PElec, 1);
  // ***Assumes field aligned with z-axis***
  m_fields[ne_idx]->PhysDeriv(2, PElec, parGradPElec);
  Vmath::Vsub(npts, outarray[Ge_idx], 1, parGradPElec, 1, outarray[Ge_idx], 1);

  // Subtract parallel pressure gradient for Ions from outarray[Ge_idx]
  // N.B. ne === nd
  Array<OneD, NekDouble> PIons(npts), parGradPIons(npts);
  Vmath::Smul(npts, m_Td, inarray[ne_idx], 1, PIons, 1);
  // ***Assumes field aligned with z-axis***
  m_fields[ne_idx]->PhysDeriv(2, PIons, parGradPIons);
  Vmath::Vsub(npts, outarray[Gd_idx], 1, parGradPIons, 1, outarray[Gd_idx], 1);
}

void LAPDSystem::CalcCollisionFreqs(const Array<OneD, NekDouble> &ne,
                                    Array<OneD, NekDouble> &nu_ei) {
  Array<OneD, NekDouble> logLambda(ne.size());
  CalcCoulombLogarithm(ne, logLambda);
  for (auto ii = 0; ii < ne.size(); ii++) {
    nu_ei[ii] = m_nu_ei_const * ne[ii] * logLambda[ii];
  }
}

void LAPDSystem::CalcCoulombLogarithm(const Array<OneD, NekDouble> &ne,
                                      Array<OneD, NekDouble> &LogLambda) {
  /* logLambda = m_coulomb_log_const - 0.5\ln n_e
       where:
         m_coulomb_log_const = 30 − \ln Z_i +1.5\ln T_e
         n_e in SI units
  */
  for (auto ii = 0; ii < LogLambda.size(); ii++) {
    LogLambda[ii] = m_coulomb_log_const - 0.5 * std::log(m_n_to_SI * ne[ii]);
  }
}

/**
 * @brief Compute E = \f$ -\nabla\phi\f$, \f$ v_{E\times B}\f$ and the
 advection
 * velocities used in the ne/Ge, Gd equations.
 * @param inarray array of field physvals
 */
void LAPDSystem::CalcEAndAdvVels(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray) {
  DriftReducedSystem::CalcEAndAdvVels(inarray);
  int nPts = GetNpoints();

  int ne_idx = m_field_to_index.get_idx("ne");
  int Gd_idx = m_field_to_index.get_idx("Gd");
  int Ge_idx = m_field_to_index.get_idx("Ge");

  // v_par,d = Gd / max(ne,n_floor) / md   (N.B. ne === nd)
  for (auto ii = 0; ii < nPts; ii++) {
    m_vParIons[ii] = inarray[Gd_idx][ii] /
                     std::max(inarray[ne_idx][ii], m_nRef * m_n_floor_fac);
  }
  Vmath::Smul(nPts, 1.0 / m_md, m_vParIons, 1, m_vParIons, 1);

  // v_par,e = Ge / max(ne,n_floor) / me
  for (auto ii = 0; ii < nPts; ii++) {
    m_vParElec[ii] = inarray[Ge_idx][ii] /
                     std::max(inarray[ne_idx][ii], m_nRef * m_n_floor_fac);
  }
  Vmath::Smul(nPts, 1.0 / m_me, m_vParElec, 1, m_vParElec, 1);

  /*
  Store difference in parallel velocities in m_vAdvDiffPar
  N.B. Outer dimension of storage has size ndim to allow it to be used in
  advection operation later
  */
  Vmath::Vsub(nPts, m_vParElec, 1, m_vParIons, 1, m_vAdvDiffPar[2], 1);

  // vAdv[iDim] = b[iDim]*v_par + v_ExB[iDim] for each species
  for (auto iDim = 0; iDim < m_graph->GetSpaceDimension(); iDim++) {
    Vmath::Svtvp(nPts, m_b_unit[iDim], m_vParElec, 1, m_vExB[iDim], 1,
                 m_vAdvElec[iDim], 1);
    Vmath::Svtvp(nPts, m_b_unit[iDim], m_vParIons, 1, m_vExB[iDim], 1,
                 m_vAdvIons[iDim], 1);
  }
}

void LAPDSystem::ExplicitTimeInt(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {

  // Zero outarray
  for (auto ifld = 0; ifld < outarray.size(); ifld++) {
    Vmath::Zero(outarray[ifld].size(), outarray[ifld], 1);
  }

  // Solver for electrostatic potential.
  SolvePhi(inarray);

  // Calculate electric field from Phi, as well as corresponding velocities for
  // all advection operations
  CalcEAndAdvVels(inarray);

  // Add advection terms to outarray, handling (ne, Ge), Gd and w separately
  AddAdvTerms({"ne", "Ge"}, m_advElec, m_vAdvElec, inarray, outarray, time);
  AddAdvTerms({"Gd"}, m_advIons, m_vAdvIons, inarray, outarray, time);
  AddAdvTerms({"w"}, m_advVort, m_vExB, inarray, outarray, time);

  AddGradPTerms(inarray, outarray);

  AddEParTerms(inarray, outarray);

  // Add collision terms to RHS of Ge, Gd eqns
  AddCollisionTerms(inarray, outarray);
  // Add polarisation drift term to vorticity eqn RHS
  AddAdvTerms({"ne"}, m_advPD, m_vAdvDiffPar, inarray, outarray, time, {"w"});

  // Add density source via xml-defined function
  AddDensitySource(outarray);
}

/**
 * @brief Compute the flux vector for advection in the ion momentum equation.
 *
 * @param physfield   Array of Fields ptrs
 * @param flux        Resulting flux array
 */
void LAPDSystem::GetFluxVectorIons(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  GetFluxVector(physfield, m_vAdvIons, flux);
}

void LAPDSystem::GetFluxVectorPD(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  GetFluxVector(physfield, m_vAdvDiffPar, flux);
}

// Set rhs = w * B^2 / (m_d * m_nRef)
void LAPDSystem::GetPhiSolveRHS(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, NekDouble> &rhs) {

  int nPts = GetNpoints();
  int w_idx = m_field_to_index.get_idx("w");
  Vmath::Smul(nPts, m_Bmag * m_Bmag / m_nRef / m_md, inarray[w_idx], 1, rhs, 1);
}

/**
 * @brief Compute the normal advection velocity for the ion momentum equation
 */
Array<OneD, NekDouble> &LAPDSystem::GetVnAdvIons() {
  return GetVnAdv(m_traceVnIons, m_vAdvIons);
}

Array<OneD, NekDouble> &LAPDSystem::GetVnAdvPD() {
  return GetVnAdv(m_traceVnPD, m_vAdvDiffPar);
}

void LAPDSystem::LoadParams() {
  DriftReducedSystem::LoadParams();

  // Factor to convert densities back to SI; used in the Coulomb logarithm calc
  m_session->LoadParameter("ns", m_n_to_SI, 1.0);

  // Charge
  m_session->LoadParameter("e", m_charge_e, 1.0);

  // Ion mass
  m_session->LoadParameter("md", m_md, 2.0);

  // Electron mass - default val is multiplied by 60 to improve convergence
  m_session->LoadParameter("me", m_me, 60. / 1836);

  // Electron temperature in eV
  m_session->LoadParameter("Te", m_Te, 5.0);

  // Ion temperature in eV
  m_session->LoadParameter("Td", m_Td, 0.1);

  // Density independent part of the coulomb logarithm
  m_session->LoadParameter("logLambda_const", m_coulomb_log_const);

  // Pre-factor used when calculating collision frequencies; read from config
  m_session->LoadParameter("nu_ei_const", m_nu_ei_const);
}

/**
 * @brief Initialization object for LAPDSystem class.
 */
void LAPDSystem::v_InitObject(bool DeclareField) {
  DriftReducedSystem::v_InitObject(DeclareField);
  // Create storage for advection velocities, parallel velocity difference, ExB
  // drift velocity, E field
  int nPts = GetNpoints();
  for (int i = 0; i < m_graph->GetSpaceDimension(); ++i) {
    m_vAdvIons[i] = Array<OneD, NekDouble>(nPts);
    m_vAdvDiffPar[i] = Array<OneD, NekDouble>(nPts);
    Vmath::Zero(nPts, m_vAdvDiffPar[i], 1);
  }
  // Create storage for ion parallel velocities
  m_vParIons = Array<OneD, NekDouble>(nPts);

  // Define the normal velocity fields.
  // These are populated at each step (by reference) in calls to GetVnAdv()
  if (m_fields[0]->GetTrace()) {
    auto nTrace = GetTraceNpoints();
    m_traceVnIons = Array<OneD, NekDouble>(nTrace);
    m_traceVnPD = Array<OneD, NekDouble>(nTrace);
  }

  // Advection objects
  m_advIons =
      SolverUtils::GetAdvectionFactory().CreateInstance(m_advType, m_advType);
  m_advPD =
      SolverUtils::GetAdvectionFactory().CreateInstance(m_advType, m_advType);

  // Set callback functions to compute flux vectors
  m_advIons->SetFluxVector(&LAPDSystem::GetFluxVectorIons, this);
  m_advPD->SetFluxVector(&LAPDSystem::GetFluxVectorPD, this);

  // Create Riemann solvers (one per advection object) and set normal  velocity
  // callback functions
  m_riemannSolverIons = SolverUtils::GetRiemannSolverFactory().CreateInstance(
      m_RiemSolvType, m_session);
  m_riemannSolverIons->SetScalar("Vn", &LAPDSystem::GetVnAdvIons, this);
  m_riemannSolverPD = SolverUtils::GetRiemannSolverFactory().CreateInstance(
      m_RiemSolvType, m_session);
  m_riemannSolverPD->SetScalar("Vn", &LAPDSystem::GetVnAdvPD, this);

  // Tell advection objects about the Riemann solvers and finish init
  m_advIons->SetRiemannSolver(m_riemannSolverIons);
  m_advIons->InitObject(m_session, m_fields);
  m_advPD->InitObject(m_session, m_fields);
  m_advPD->SetRiemannSolver(m_riemannSolverPD);

  // Bind RHS function for time integration object
  m_ode.DefineOdeRhs(&LAPDSystem::ExplicitTimeInt, this);
}

} // namespace Nektar
