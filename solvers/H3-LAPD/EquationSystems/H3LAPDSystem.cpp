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

#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

#include "H3LAPDSystem.h"

namespace Nektar {
std::string H3LAPDSystem::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "LAPD", H3LAPDSystem::create, "Hermes-3 LAPD equation system");

H3LAPDSystem::H3LAPDSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
                           const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), AdvectionSystem(pSession, pGraph),
      m_field_to_index(pSession->GetVariables()) {
  m_required_flds = {"n_e", "p_d", "p_e", "w"};
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
  // Ensure that the session file defines all required variables
  ValidateFieldList();

  AdvectionSystem::v_InitObject(DeclareField);

  // // Populate m_intVariables

  // // Since we are starting from a setup where each field is defined to be
  // // a discontinuous field (and thus support DG), the first thing we do is
  // // to recreate the phi field so that it is continuous, in order to support
  // the
  // // Poisson solve. Note that you can still perform a Poisson solve using
  // // a discontinuous field, which is done via the hybridisable
  // // discontinuous Galerkin (HDG) approach.
  // int phi_idx = nVar - 1;
  // m_fields[phi_idx] =
  // MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
  //     m_session, m_graph, m_session->GetVariable(phi_idx), true, true);

  // // Assign storage for drift velocity.
  // for (int i = 0; i < 2; ++i) {
  //   m_driftVel[i] = Array<OneD, NekDouble>(nPts);
  // }

  // // Type of advection class to be used. By default, we only support the
  // // discontinuous projection, since this is the only approach we're
  // // considering for this solver.
  // ASSERTL0(m_projectionType == MultiRegions::eDiscontinuous,
  //          "Unsupported projection type: only discontinuous"
  //          " projection supported.");

  // // Do not forwards transform initial condition.
  // m_homoInitialFwd = false;

  // // Define the normal velocity fields.
  // if (m_fields[0]->GetTrace()) {
  //   m_traceVn = Array<OneD, NekDouble>(GetTraceNpoints());
  // }

  // // The remainder of this code is fairly generic boilerplate for the DG
  // // setup.
  // std::string advName, riemName;

  // // Load what type of advection we want to use -- in theory we also
  // // support flux reconstruction for quad-based meshes, or you can use a
  // // standard convective term if you were fully continuous in
  // // space. Default is DG.
  // m_session->LoadSolverInfo("AdvectionType", advName, "WeakDG");

  // // Create an advection object of the type above using the
  // // factory pattern.
  // m_advObject =
  //     SolverUtils::GetAdvectionFactory().CreateInstance(advName, advName);

  // // The advection object needs to know the flux vector being calculated:
  // // this is done with a callback.
  // m_advObject->SetFluxVector(&H3LAPDSystem::GetFluxVector, this);

  // // Repeat the above for the Riemann solver: in this case we use an
  // // upwind by default. The solver also needs to know the trace normal,
  // // which we again implement using a callback.
  // m_session->LoadSolverInfo("UpwindType", riemName, "Upwind");
  // m_riemannSolver =
  //     GetRiemannSolverFactory().CreateInstance(riemName, m_session);
  // m_riemannSolver->SetScalar("Vn", &H3LAPDSystem::GetNormalVelocity, this);

  // // Tell the advection object about the Riemann solver to use,
  // // and then get it set up.
  // m_advObject->SetRiemannSolver(m_riemannSolver);
  // m_advObject->InitObject(m_session, m_fields);

  // ASSERTL0(m_explicitAdvection,
  //          "This solver only supports explicit-in-time advection.");
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
void DoOdeProjection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                     Array<OneD, Array<OneD, NekDouble>> &outarray,
                     const NekDouble time) {
  int nvariables = inarray.size();
  int npoints = inarray[0].size();
  // SetBoundaryConditions(time);

  for (int i = 0; i < nvariables; ++i) {
    Vmath::Vcopy(npoints, inarray[i], 1, outarray[i], 1);
  }
}

/**
 * @brief Compute the flux vector for this system.
 *
 * @param physfield   Array of Fields ptrs
 * @param flux        Resulting flux array
 */
void H3LAPDSystem::GetFluxVector(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {}

/**
 * @brief Return the flux vector for the diffusion problem.
 */
void H3LAPDSystem::GetFluxVectorDiff(
    const Array<OneD, Array<OneD, NekDouble>> &inarray,
    const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &qfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscousTensor) {}

} // namespace Nektar
