///////////////////////////////////////////////////////////////////////////////
//
// File H3LAPDSystem.cpp
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
// Description: Hermes-3 LAPD equation system
//
///////////////////////////////////////////////////////////////////////////////
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

#include "H3LAPDSystem.hpp"

namespace Nektar {
std::string H3LAPDSystem::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "LAPD", H3LAPDSystem::create, "Hermes-3 LAPD equation system");

H3LAPDSystem::H3LAPDSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
                           const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), AdvectionSystem(pSession, pGraph),
      m_field_to_index(pSession->GetVariables()),
      m_vAdvDiffPar(pGraph->GetSpaceDimension()),
      m_vAdvElec(pGraph->GetSpaceDimension()),
      m_vAdvIons(pGraph->GetSpaceDimension()),
      m_vExB(pGraph->GetSpaceDimension()), m_E(pGraph->GetSpaceDimension()) {
  m_required_flds = {"ne", "Ge", "Gd", "w", "phi"};
  m_int_fld_names = {"ne", "Ge", "Gd", "w"};
  // Construct particle system
  m_particle_sys = std::make_shared<NeutralParticleSystem>(pSession, pGraph);

  // mass recording diagnostic creation
  m_diag_mass_recording_enabled =
      pSession->DefinesParameter("mass_recording_step");
}

void H3LAPDSystem::AddAdvTerms(
    std::vector<std::string> field_names,
    const SolverUtils::AdvectionSharedPtr advObj,
    const Array<OneD, Array<OneD, NekDouble>> &vAdv,
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time,
    std::vector<std::string> eqn_labels) {

  // Default is to add result of advecting field f to the RHS of df/dt equation
  if (eqn_labels.empty()) {
    eqn_labels = std::vector(field_names);
  } else {
    ASSERTL1(field_names.size() == eqn_labels.size(),
             "AddAdvTerms: Number of quantities being advected must match the "
             "number of equation labels.");
  }

  int nfields = field_names.size();
  int npts = inarray[0].size();

  // Make temporary copies of target fields, inarray vals and initialise a
  // temporary output array
  Array<OneD, MultiRegions::ExpListSharedPtr> tmp_fields(nfields);
  Array<OneD, Array<OneD, NekDouble>> tmp_inarray(nfields);
  Array<OneD, Array<OneD, NekDouble>> tmp_outarray(nfields);
  for (auto ii = 0; ii < nfields; ii++) {
    int idx = m_field_to_index.get_idx(field_names[ii]);
    tmp_fields[ii] = m_fields[idx];
    tmp_inarray[ii] = Array<OneD, NekDouble>(npts);
    Vmath::Vcopy(npts, inarray[idx], 1, tmp_inarray[ii], 1);
    tmp_outarray[ii] = Array<OneD, NekDouble>(outarray[idx].size());
  }
  // Compute advection terms; result is returned in temporary output array
  advObj->Advect(tmp_fields.size(), tmp_fields, vAdv, tmp_inarray, tmp_outarray,
                 time);

  // Subtract temporary output array from the appropriate indices of outarray
  for (auto ii = 0; ii < nfields; ii++) {
    int idx = m_field_to_index.get_idx(eqn_labels[ii]);
    Vmath::Vsub(outarray[idx].size(), outarray[idx], 1, tmp_outarray[ii], 1,
                outarray[idx], 1);
  }
}

void H3LAPDSystem::AddCollisionTerms(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray) {

  int npts = inarray[0].size();

  // Field indices
  int ne_idx = m_field_to_index.get_idx("ne");
  int Ge_idx = m_field_to_index.get_idx("Ge");
  int Gd_idx = m_field_to_index.get_idx("Gd");
  int w_idx = m_field_to_index.get_idx("w");

  // Calculate collision term
  // This is the momentum(-density) tranferred from electrons to ions by
  // collisions, so add it to Gd rhs, but subtract it from Ge rhs
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

void H3LAPDSystem::AddDensitySource(
    Array<OneD, Array<OneD, NekDouble>> &outarray) {

  int ne_idx = m_field_to_index.get_idx("ne");
  int nPts = GetNpoints();
  Array<OneD, NekDouble> tmpx(nPts), tmpy(nPts), tmpz(nPts);
  m_fields[ne_idx]->GetCoords(tmpx, tmpy, tmpz);
  Array<OneD, NekDouble> dens_src(nPts, 0.0);
  LibUtilities::EquationSharedPtr dens_src_func =
      m_session->GetFunction("dens_src", ne_idx);
  dens_src_func->Evaluate(tmpx, tmpy, tmpz, dens_src);
  Vmath::Vadd(nPts, outarray[ne_idx], 1, dens_src, 1, outarray[ne_idx], 1);
}

void H3LAPDSystem::AddEParTerms(
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

void H3LAPDSystem::AddGradPTerms(
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
  // ne === nd
  Array<OneD, NekDouble> PIons(npts), parGradPIons(npts);
  Vmath::Smul(npts, m_Td, inarray[ne_idx], 1, PIons, 1);
  // ***Assumes field aligned with z-axis***
  m_fields[ne_idx]->PhysDeriv(2, PIons, parGradPIons);
  Vmath::Vsub(npts, outarray[Gd_idx], 1, parGradPIons, 1, outarray[Gd_idx], 1);
}

/**
 *  Called from ExplicitTimeInt() to add particle sources (stored in "*_src"
 * fields) to the RHS array.
 *
 *  @param outarray the RHS array
 *
 */
void H3LAPDSystem::AddParticleSources(
    std::vector<std::string> target_fields,
    Array<OneD, Array<OneD, NekDouble>> &outarray) {
  for (auto target_field : target_fields) {
    int src_field_idx = m_field_to_index.get_idx(target_field + "_src");

    if (src_field_idx >= 0) {
      auto tmp_it = std::find(m_int_fld_names.cbegin(), m_int_fld_names.cend(),
                              target_field);
      ASSERTL0(tmp_it != m_int_fld_names.cend(),
               "Target field for particle source ['" + target_field +
                   "'] term not recognised.")
      auto field_idx = std::distance(m_int_fld_names.cbegin(), tmp_it);
      Vmath::Vadd(outarray[field_idx].size(), outarray[field_idx], 1,
                  m_fields[src_field_idx]->GetPhys(), 1, outarray[field_idx],
                  1);
    }
  }
}

void H3LAPDSystem::CalcCollisionFreqs(const Array<OneD, NekDouble> &ne,
                                      Array<OneD, NekDouble> &nu_ei) {
  Array<OneD, NekDouble> logLambda(ne.size());
  CalcCoulombLogarithm(ne, logLambda);
  for (auto ii = 0; ii < ne.size(); ii++) {
    nu_ei[ii] = m_nu_ei_const * ne[ii] * logLambda[ii];
  }
}

void H3LAPDSystem::CalcCoulombLogarithm(const Array<OneD, NekDouble> &ne,
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
 * @brief Compute E = \f$ -\nabla\phi\f$, \f$ v_{E\times B}\f$ and the advection
 * velocities used in the ne/Ge, Gd equations.
 * @param inarray array of field physvals
 */
void H3LAPDSystem::CalcEAndAdvVels(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray) {
  int phi_idx = m_field_to_index.get_idx("phi");
  int nPts = GetNpoints();
  m_fields[phi_idx]->PhysDeriv(m_fields[phi_idx]->GetPhys(), m_E[0], m_E[1],
                               m_E[2]);
  Vmath::Neg(nPts, m_E[0], 1);
  Vmath::Neg(nPts, m_E[1], 1);
  Vmath::Neg(nPts, m_E[2], 1);

  // v_ExB = Evec x Bvec / B^2
  Vmath::Svtsvtp(nPts, m_B[2] / m_Bmag / m_Bmag, m_E[1], 1,
                 -m_B[1] / m_Bmag / m_Bmag, m_E[2], 1, m_vExB[0], 1);
  Vmath::Svtsvtp(nPts, m_B[0] / m_Bmag / m_Bmag, m_E[2], 1,
                 -m_B[2] / m_Bmag / m_Bmag, m_E[0], 1, m_vExB[1], 1);
  Vmath::Svtsvtp(nPts, m_B[1] / m_Bmag / m_Bmag, m_E[0], 1,
                 -m_B[0] / m_Bmag / m_Bmag, m_E[1], 1, m_vExB[2], 1);

  int ne_idx = m_field_to_index.get_idx("ne");
  // If we're evolving ion momentum, use it to set the ion parallel velocity.
  // Otherwise zero the array
  int Gd_idx = m_field_to_index.get_idx("Gd");
  if (Gd_idx >= 0) {
    // v_par,d = Gd / max(ne,n_floor) / md   (N.B. ne === nd)
    for (auto ii = 0; ii < nPts; ii++) {
      m_vParIons[ii] = inarray[Gd_idx][ii] /
                       std::max(inarray[ne_idx][ii], m_nRef * m_n_floor_fac);
    }
    Vmath::Smul(nPts, 1.0 / m_md, m_vParIons, 1, m_vParIons, 1);
  } else {
    Vmath::Zero(nPts, m_vParIons, 1);
  }

  // If we're evolving electron momentum, use it to set the electron parallel
  // velocity. Otherwise zero the array
  if (Gd_idx >= 0) {
    // v_par,e = Ge / max(ne,n_floor) / me
    int Ge_idx = m_field_to_index.get_idx("Ge");
    for (auto ii = 0; ii < nPts; ii++) {
      m_vParElec[ii] = inarray[Ge_idx][ii] /
                       std::max(inarray[ne_idx][ii], m_nRef * m_n_floor_fac);
    }
    Vmath::Smul(nPts, 1.0 / m_me, m_vParElec, 1, m_vParElec, 1);
  } else {
    Vmath::Zero(nPts, m_vParElec, 1);
  }

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

/**
 * @brief Perform projection into correct polynomial space.
 *
 * This routine projects the @p inarray input and ensures the @p outarray
 * output lives in the correct space. Since we are hard-coding DG, this
 * corresponds to a simple copy from in to out, since no elemental
 * connectivity is required and the output of the RHS function is
 * polynomial.
 */
void H3LAPDSystem::DoOdeProjection(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {
  int nvariables = inarray.size();
  int npoints = inarray[0].size();
  // SetBoundaryConditions(time);

  for (int i = 0; i < nvariables; ++i) {
    Vmath::Vcopy(npoints, inarray[i], 1, outarray[i], 1);
  }
}

void H3LAPDSystem::ExplicitTimeInt(
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

void GetFluxVector(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                   const Array<OneD, Array<OneD, NekDouble>> &vAdv,
                   Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  ASSERTL1(flux[0].size() == vAdv.size(),
           "Dimension of flux array and advection velocity array do not match");
  int nq = physfield[0].size();

  for (auto i = 0; i < flux.size(); ++i) {
    for (auto j = 0; j < flux[0].size(); ++j) {
      Vmath::Vmul(nq, physfield[i], 1, vAdv[j], 1, flux[i][j], 1);
    }
  }
}

/**
 * @brief Return the flux vector for the diffusion problem.
 */
void H3LAPDSystem::GetFluxVectorDiff(
    const Array<OneD, Array<OneD, NekDouble>> &inarray,
    const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &qfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscousTensor) {
  std::cout << "*** GetFluxVectorDiff not defined! ***" << std::endl;
}

/**
 * @brief Compute the flux vector for advection in the electron density and
 * momentum equations.
 *
 * @param physfield   Array of Fields ptrs
 * @param flux        Resulting flux array
 */
void H3LAPDSystem::GetFluxVectorElec(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  GetFluxVector(physfield, m_vAdvElec, flux);
}

/**
 * @brief Compute the flux vector for advection in the ion momentum equation.
 *
 * @param physfield   Array of Fields ptrs
 * @param flux        Resulting flux array
 */
void H3LAPDSystem::GetFluxVectorIons(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  GetFluxVector(physfield, m_vAdvIons, flux);
}

void H3LAPDSystem::GetFluxVectorPD(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  GetFluxVector(physfield, m_vAdvDiffPar, flux);
}

/**
 * @brief Compute the flux vector for advection in the vorticity equation.
 *
 * @param physfield   Array of Fields ptrs
 * @param flux        Resulting flux array
 */
void H3LAPDSystem::GetFluxVectorVort(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  // Advection velocity is v_ExB in the vorticity equation
  GetFluxVector(physfield, m_vExB, flux);
}

// Set rhs = w * B^2 / (m_d * m_nRef)
void H3LAPDSystem::GetPhiSolveRHS(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, NekDouble> &rhs) {

  int nPts = GetNpoints();
  int w_idx = m_field_to_index.get_idx("w");
  Vmath::Smul(nPts, m_Bmag * m_Bmag / m_nRef / m_md, inarray[w_idx], 1, rhs, 1);
}

/**
 * @brief Compute normal advection velocity given a trace array and an advection
 * velocity array
 */
Array<OneD, NekDouble> &
H3LAPDSystem::GetVnAdv(Array<OneD, NekDouble> &traceVn,
                       const Array<OneD, Array<OneD, NekDouble>> &vAdv) {
  // Number of trace (interface) points
  int nTracePts = GetTraceNpoints();
  // Auxiliary variable to compute normal velocities
  Array<OneD, NekDouble> tmp(nTracePts);

  // Zero previous values
  Vmath::Zero(nTracePts, traceVn, 1);

  //  Compute dot product of advection velocity with the trace normals and store
  for (int i = 0; i < vAdv.size(); ++i) {
    m_fields[0]->ExtractTracePhys(vAdv[i], tmp);
    Vmath::Vvtvp(nTracePts, m_traceNormals[i], 1, tmp, 1, traceVn, 1, traceVn,
                 1);
  }
  return traceVn;
}

/**
 * @brief Compute the normal advection velocity for the electron density
 */
Array<OneD, NekDouble> &H3LAPDSystem::GetVnAdvElec() {
  return GetVnAdv(m_traceVnElec, m_vAdvElec);
}

/**
 * @brief Compute the normal advection velocity for the ion momentum equation
 */
Array<OneD, NekDouble> &H3LAPDSystem::GetVnAdvIons() {
  return GetVnAdv(m_traceVnIons, m_vAdvIons);
}

Array<OneD, NekDouble> &H3LAPDSystem::GetVnAdvPD() {
  return GetVnAdv(m_traceVnPD, m_vAdvDiffPar);
}

/**
 * @brief Compute the normal advection velocity for the vorticity equation
 */
Array<OneD, NekDouble> &H3LAPDSystem::GetVnAdvVort() {
  return GetVnAdv(m_traceVnVort, m_vExB);
}

void H3LAPDSystem::LoadParams() {
  // Type of advection to use -- in theory we also support flux reconstruction
  // for quad-based meshes, or you can use a standard convective term if you
  // were fully continuous in space. Default is DG.
  m_session->LoadSolverInfo("AdvectionType", m_advType, "WeakDG");

  // ***Assumes field aligned with z-axis***
  // Magnetic field strength. Fix B = [0, 0, Bxy] for now
  m_B = std::vector<NekDouble>(m_graph->GetSpaceDimension(), 0);
  m_session->LoadParameter("Bxy", m_B[2], 0.1);

  // Coefficient factors for potential solve
  m_session->LoadParameter("d00", m_d00, 1);
  m_session->LoadParameter("d11", m_d11, 1);
  m_session->LoadParameter("d22", m_d22, 1);

  // Factor to set density floor; default to 1e-5 (Hermes-3 default)
  m_session->LoadParameter("n_floor_fac", m_n_floor_fac, 1e-5);

  // Factor to convert densities back to SI; used in the Coulomb logarithm calc
  m_session->LoadParameter("ns", m_n_to_SI, 1.0);

  // Charge
  m_session->LoadParameter("e", m_charge_e, 1.0);

  // Ion mass
  m_session->LoadParameter("md", m_md, 2.0);

  // Electron mass - default val is multiplied by 60 to improve convergence
  m_session->LoadParameter("me", m_me, 60. / 1836);

  // Reference number density
  m_session->LoadParameter("nRef", m_nRef, 1.0);

  // Electron temperature in eV
  m_session->LoadParameter("Te", m_Te, 5.0);

  // Ion temperature in eV
  m_session->LoadParameter("Td", m_Td, 0.1);

  // Type of Riemann solver to use. Default = "Upwind"
  m_session->LoadSolverInfo("UpwindType", m_RiemSolvType, "Upwind");

  // Don't try to read no-default params for eqn sys types that don't use them
  std::string eq_sys_name = m_session->GetSolverInfo("EQTYPE");
  if (eq_sys_name != "HWLAPD") {
    // Density independent part of the coulomb logarithm
    m_session->LoadParameter("logLambda_const", m_coulomb_log_const);

    // Pre-factor used when calculating collision frequencies; read from config
    m_session->LoadParameter("nu_ei_const", m_nu_ei_const);
  }

  // Particle-related parameters
  m_session->LoadParameter("num_particle_steps_per_fluid_step",
                           m_num_part_substeps, 1);
  m_session->LoadParameter("particle_num_write_particle_steps",
                           m_num_write_particle_steps, 0);
  m_part_timestep = m_timestep / m_num_part_substeps;
}

void H3LAPDSystem::PrintArrSize(const Array<OneD, NekDouble> &arr,
                                std::string label, bool all_tasks) {
  if (m_session->GetComm()->TreatAsRankZero() || all_tasks) {
    if (!label.empty()) {
      std::cout << label << " ";
    }
    std::cout << "size = " << arr.size() << std::endl;
  }
}

void H3LAPDSystem::PrintArrVals(const Array<OneD, NekDouble> &arr, int num,
                                int stride, std::string label, bool all_tasks) {
  if (m_session->GetComm()->TreatAsRankZero() || all_tasks) {
    if (!label.empty()) {
      std::cout << "[" << label << "]" << std::endl;
    }
    int ii_max = std::min(static_cast<int>(arr.size()), num * stride);
    for (auto ii = 0; ii < ii_max; ii = ii + stride) {
      std::cout << "  " << std::setprecision(12) << arr[ii] << std::endl;
    }
  }
}

void H3LAPDSystem::SolvePhi(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray) {

  // Field indices
  int nPts = GetNpoints();
  int phi_idx = m_field_to_index.get_idx("phi");

  // Define rhs
  Array<OneD, NekDouble> rhs(nPts);
  GetPhiSolveRHS(inarray, rhs);

  // Set up factors for electrostatic potential solve
  StdRegions::ConstFactorMap factors;
  // Helmholtz => Poisson (lambda = 0)
  factors[StdRegions::eFactorLambda] = 0.0;
  // Set coefficient factors
  factors[StdRegions::eFactorCoeffD00] = m_d00;
  factors[StdRegions::eFactorCoeffD11] = m_d11;
  factors[StdRegions::eFactorCoeffD22] = m_d22;

  // Solve for phi. Output of this routine is in coefficient (spectral)
  // space, so backwards transform to physical space since we'll need that
  // for the advection step & computing drift velocity.
  m_fields[phi_idx]->HelmSolve(rhs, m_fields[phi_idx]->UpdateCoeffs(), factors);
  m_fields[phi_idx]->BwdTrans(m_fields[phi_idx]->GetCoeffs(),
                              m_fields[phi_idx]->UpdatePhys());
}

/**
 * Check all required fields are defined
 */
void H3LAPDSystem::ValidateFieldList() {
  for (auto &fld_name : m_required_flds) {
    ASSERTL0(m_field_to_index.get_idx(fld_name) >= 0,
             "Required field [" + fld_name + "] is not defined.");
  }
}

/**
 * @brief Initialization object for H3LAPDSystem class.
 */
void H3LAPDSystem::v_InitObject(bool DeclareField) {
  //  Ensure that the session file defines all required variables
  ValidateFieldList();

  AdvectionSystem::v_InitObject(DeclareField);

  // Load parameters
  LoadParams();

  // Compute some properties derived from params
  m_Bmag = std::sqrt(m_B[0] * m_B[0] + m_B[1] * m_B[1] + m_B[2] * m_B[2]);
  m_b_unit = std::vector<NekDouble>(m_graph->GetSpaceDimension());
  for (auto idim = 0; idim < m_b_unit.size(); idim++) {
    m_b_unit[idim] = m_B[idim] / m_Bmag;
  }

  // Tell UnsteadySystem to only integrate a subset of fields in time
  // (Ignore fields that don't have a time derivative)
  m_intVariables.resize(m_int_fld_names.size());
  for (auto ii = 0; ii < m_int_fld_names.size(); ii++) {
    int var_idx = m_field_to_index.get_idx(m_int_fld_names[ii]);
    ASSERTL0(var_idx >= 0, "Setting time integration vars - GetIntFieldNames() "
                           "returned an invalid field name.");
    m_intVariables[ii] = var_idx;
  }

  // Since we are starting from a setup where each field is defined to be a
  // discontinuous field (and thus support DG), the first thing we do is to
  // recreate the phi field so that it is continuous, in order to support the
  // Poisson solve. Note that you can still perform a Poisson solve using a
  // discontinuous field, which is done via the hybridisable discontinuous
  // Galerkin (HDG) approach.
  int phi_idx = m_field_to_index.get_idx("phi");
  m_fields[phi_idx] = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
      m_session, m_graph, m_session->GetVariable(phi_idx), true, true);

  // Create storage for advection velocities, parallel velocity difference,ExB
  // drift velocity, E field
  int nPts = GetNpoints();
  for (int i = 0; i < m_graph->GetSpaceDimension(); ++i) {
    m_vAdvElec[i] = Array<OneD, NekDouble>(nPts);
    m_vAdvIons[i] = Array<OneD, NekDouble>(nPts);
    m_vAdvDiffPar[i] = Array<OneD, NekDouble>(nPts);
    Vmath::Zero(nPts, m_vAdvDiffPar[i], 1);
    m_vExB[i] = Array<OneD, NekDouble>(nPts);
    m_E[i] = Array<OneD, NekDouble>(nPts);
  }
  // Create storage for parallel velocities
  m_vParIons = Array<OneD, NekDouble>(nPts);
  m_vParElec = Array<OneD, NekDouble>(nPts);

  // Type of advection class to be used. By default, we only support the
  // discontinuous projection, since this is the only approach we're
  // considering for this solver.
  ASSERTL0(m_projectionType == MultiRegions::eDiscontinuous,
           "Unsupported projection type: only discontinuous"
           " projection supported."); ////

  // Do not forwards transform initial condition.
  m_homoInitialFwd = false; ////

  // Define the normal velocity fields.
  // These are populated at each step (by reference) in calls to GetVnAdv()
  if (m_fields[0]->GetTrace()) {
    auto nTrace = GetTraceNpoints();
    m_traceVnElec = Array<OneD, NekDouble>(nTrace);
    m_traceVnIons = Array<OneD, NekDouble>(nTrace);
    m_traceVnPD = Array<OneD, NekDouble>(nTrace);
    m_traceVnVort = Array<OneD, NekDouble>(nTrace);
  }

  // Advection objects
  // Need one per advection velocity
  m_advElec =
      SolverUtils::GetAdvectionFactory().CreateInstance(m_advType, m_advType);
  m_advIons =
      SolverUtils::GetAdvectionFactory().CreateInstance(m_advType, m_advType);
  m_advPD =
      SolverUtils::GetAdvectionFactory().CreateInstance(m_advType, m_advType);
  m_advVort =
      SolverUtils::GetAdvectionFactory().CreateInstance(m_advType, m_advType);

  // Set callback functions to compute flux vectors
  m_advElec->SetFluxVector(&H3LAPDSystem::GetFluxVectorElec, this);
  m_advIons->SetFluxVector(&H3LAPDSystem::GetFluxVectorIons, this);
  m_advPD->SetFluxVector(&H3LAPDSystem::GetFluxVectorPD, this);
  m_advVort->SetFluxVector(&H3LAPDSystem::GetFluxVectorVort, this);

  // Create Riemann solvers (one per advection object) and set normal velocity
  // callback functions
  m_riemannSolverElec = SolverUtils::GetRiemannSolverFactory().CreateInstance(
      m_RiemSolvType, m_session);
  m_riemannSolverElec->SetScalar("Vn", &H3LAPDSystem::GetVnAdvElec, this);
  m_riemannSolverIons = SolverUtils::GetRiemannSolverFactory().CreateInstance(
      m_RiemSolvType, m_session);
  m_riemannSolverIons->SetScalar("Vn", &H3LAPDSystem::GetVnAdvIons, this);
  m_riemannSolverPD = SolverUtils::GetRiemannSolverFactory().CreateInstance(
      m_RiemSolvType, m_session);
  m_riemannSolverPD->SetScalar("Vn", &H3LAPDSystem::GetVnAdvPD, this);
  m_riemannSolverVort = SolverUtils::GetRiemannSolverFactory().CreateInstance(
      m_RiemSolvType, m_session);
  m_riemannSolverVort->SetScalar("Vn", &H3LAPDSystem::GetVnAdvVort, this);

  // Tell advection objects about the Riemann solvers and finish init
  m_advElec->SetRiemannSolver(m_riemannSolverElec);
  m_advElec->InitObject(m_session, m_fields);
  m_advIons->SetRiemannSolver(m_riemannSolverIons);
  m_advIons->InitObject(m_session, m_fields);
  m_advPD->InitObject(m_session, m_fields);
  m_advPD->SetRiemannSolver(m_riemannSolverPD);
  m_advVort->SetRiemannSolver(m_riemannSolverVort);
  m_advVort->InitObject(m_session, m_fields);

  // The m_ode object defines the timestepping to be used, and lives in
  // the SolverUtils::UnsteadySystem class. For explicit solvers, you need
  // to supply a right-hand side function, and a projection function
  // (e.g. for continuous Galerkin this would be an assembly-type
  // operation to ensure C^0 connectivity). These are done again through
  // callbacks.
  m_ode.DefineOdeRhs(&H3LAPDSystem::ExplicitTimeInt, this);
  m_ode.DefineProjection(&H3LAPDSystem::DoOdeProjection, this);

  ASSERTL0(m_explicitAdvection,
           "This solver only supports explicit-in-time advection.");

  // Store DisContFieldSharedPtr casts of fields in a map, indexed by name, for
  // use in particle project,evaluate operations
  int idx = 0;
  for (auto &field_name : m_session->GetVariables()) {
    m_discont_fields[field_name] =
        std::dynamic_pointer_cast<MultiRegions::DisContField>(m_fields[idx]);
    idx++;
  }

  // Setup object to project onto density source field
  int low_order_project;
  m_session->LoadParameter("low_order_project", low_order_project, 0);
  if (low_order_project) {
    ASSERTL0(
        m_discont_fields.count("ne_src_interp"),
        "Interediate, lower order interpolation field not found in config.");
    m_particle_sys->setup_project(m_discont_fields["ne_src_interp"],
                                  m_discont_fields["ne_src"]);
  } else {
    m_particle_sys->setup_project(m_discont_fields["ne_src"]);
  }

  // Setup object to evaluate density field
  m_particle_sys->setup_evaluate_ne(m_discont_fields["ne"]);
}

bool H3LAPDSystem::v_PostIntegrate(int step) {
  // Writes a step of the particle trajectory.
  if (m_num_write_particle_steps > 0 &&
      (step % m_num_write_particle_steps) == 0) {
    m_particle_sys->write(step);
    m_particle_sys->write_source_fields();
  }

  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recording->compute(step);
  }

  m_solver_callback_handler.call_post_integrate(this);
  return AdvectionSystem::v_PostIntegrate(step);
}

bool H3LAPDSystem::v_PreIntegrate(int step) {
  m_solver_callback_handler.call_pre_integrate(this);

  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recording->compute_initial_fluid_mass();
  }

  // Integrate the particle system to the requested time.
  m_particle_sys->integrate(m_time + m_timestep, m_part_timestep);
  // Project onto the source fields
  m_particle_sys->project_source_terms();

  return AdvectionSystem::v_PreIntegrate(step);
}

/**
 * Convenience function to zero outarray for all fields
 *
 */
void H3LAPDSystem::ZeroOutArray(Array<OneD, Array<OneD, NekDouble>> &outarray) {
  for (auto ifld = 0; ifld < outarray.size(); ifld++) {
    Vmath::Zero(outarray[ifld].size(), outarray[ifld], 1);
  }
}

} // namespace Nektar
