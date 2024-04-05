///////////////////////////////////////////////////////////////////////////////
//
// File SOLSystem.cpp
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
// Description: Equation system heavily based on CompressibleFlowSystem
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

#include "SOLSystem.h"

namespace Nektar {
std::string SOLSystem::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "SOL", SOLSystem::create, "SOL equations in conservative variables.");

SOLSystem::SOLSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
                     const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), AdvectionSystem(pSession, pGraph),
      m_field_to_index(pSession->GetVariables()) {
  m_required_flds = {"rho", "rhou", "E"};
  if (m_spacedim == 2) {
    m_required_flds.push_back("rhov");
  }
}

/**
 * Check all required fields are defined
 */
void SOLSystem::ValidateFieldList() {
  for (auto &fld_name : m_required_flds) {
    ASSERTL0(m_field_to_index.get_idx(fld_name) >= 0,
             "Required field [" + fld_name + "] is not defined.");
  }
}

/**
 * @brief Initialization object for SOLSystem class.
 */
void SOLSystem::v_InitObject(bool DeclareField) {
  ValidateFieldList();
  AdvectionSystem::v_InitObject(DeclareField);

  for (int i = 0; i < m_fields.size(); i++) {
    // Use BwdTrans to make sure initial condition is in solution space
    m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(), m_fields[i]->UpdatePhys());
  }

  m_varConv = MemoryManager<VariableConverter>::AllocateSharedPtr(m_session,
                                                                  m_spacedim);

  ASSERTL0(m_session->DefinesSolverInfo("UPWINDTYPE"),
           "No UPWINDTYPE defined in session.");

  // Set up locations of velocity vector.
  m_vecLocs = Array<OneD, Array<OneD, NekDouble>>(1);
  m_vecLocs[0] = Array<OneD, NekDouble>(m_spacedim);
  for (int i = 0; i < m_spacedim; ++i) {
    m_vecLocs[0][i] = 1 + i;
  }

  // Loading parameters from session file
  m_session->LoadParameter("Gamma", m_gamma, 1.4);

  // Setting up advection and diffusion operators
  InitAdvection();

  // Set up Forcing objects for source terms.
  m_forcing = SolverUtils::Forcing::Load(m_session, shared_from_this(),
                                         m_fields, m_fields.size());

  m_ode.DefineOdeRhs(&SOLSystem::DoOdeRhs, this);
  m_ode.DefineProjection(&SOLSystem::DoOdeProjection, this);
}

/**
 * @brief Destructor for SOLSystem class.
 */
SOLSystem::~SOLSystem() {}

/**
 * @brief Create advection and diffusion objects for CFS
 */
void SOLSystem::InitAdvection() {
  // Check if projection type is correct
  ASSERTL0(m_projectionType == MultiRegions::eDiscontinuous,
           "Unsupported projection type: must be DG.");

  std::string advName, riemName;
  m_session->LoadSolverInfo("AdvectionType", advName, "WeakDG");

  m_advObject =
      SolverUtils::GetAdvectionFactory().CreateInstance(advName, advName);

  m_advObject->SetFluxVector(&SOLSystem::GetFluxVector, this);

  // Setting up Riemann solver for advection operator
  m_session->LoadSolverInfo("UpwindType", riemName, "Average");

  SolverUtils::RiemannSolverSharedPtr riemannSolver;
  riemannSolver = SolverUtils::GetRiemannSolverFactory().CreateInstance(
      riemName, m_session);

  // Setting up parameters for advection operator Riemann solver
  riemannSolver->SetParam("gamma", &SOLSystem::GetGamma, this);
  riemannSolver->SetAuxVec("vecLocs", &SOLSystem::GetVecLocs, this);
  riemannSolver->SetVector("N", &SOLSystem::GetNormals, this);

  // Concluding initialisation of advection / diffusion operators
  m_advObject->SetRiemannSolver(riemannSolver);
  m_advObject->InitObject(m_session, m_fields);
}

/**
 * @brief Compute the right-hand side.
 */
void SOLSystem::DoOdeRhs(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {
  int nvariables = inarray.size();
  int npoints = GetNpoints();
  int nTracePts = GetTraceTotPoints();

  // Store forwards/backwards space along trace space
  Array<OneD, Array<OneD, NekDouble>> Fwd(nvariables);
  Array<OneD, Array<OneD, NekDouble>> Bwd(nvariables);

  if (m_HomogeneousType == eHomogeneous1D) {
    Fwd = NullNekDoubleArrayOfArray;
    Bwd = NullNekDoubleArrayOfArray;
  } else {
    for (int i = 0; i < nvariables; ++i) {
      Fwd[i] = Array<OneD, NekDouble>(nTracePts, 0.0);
      Bwd[i] = Array<OneD, NekDouble>(nTracePts, 0.0);
      m_fields[i]->GetFwdBwdTracePhys(inarray[i], Fwd[i], Bwd[i]);
    }
  }

  // Calculate advection
  DoAdvection(inarray, outarray, time, Fwd, Bwd);

  // Negate results
  for (int i = 0; i < nvariables; ++i) {
    Vmath::Neg(npoints, outarray[i], 1);
  }

  // // Add diffusion terms
  // DoDiffusion(inarray, outarray, Fwd, Bwd);

  // Add forcing terms
  for (auto &x : m_forcing) {
    x->Apply(m_fields, inarray, outarray, time);
  }
}

/**
 * Needs to be defined for explicit time integration, but does nothing.
 */
void SOLSystem::DoOdeProjection(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {
  // Do nothing
}

/**
 * @brief Initialises the time integration scheme (as specified in the
 * session file), and performs the time integration.
 *
 * UnsteadySystem::v_DoSolve() is duplicated more-or-less wholesale here; the
 * difference is that the field phys vals used to initialise the time
 * integrator object are *non-const* refs to those stores in m_fields.
 * Without this, changes to m_fields in other member functions will not
 * propagate to the time step loop.
 */
void SOLSystem::v_DoSolve() {
  ASSERTL0(m_intScheme != 0, "No time integration scheme.");

  int i = 1;
  int nvariables = 0;
  int nfields = m_fields.size();

  if (m_intVariables.empty()) {
    for (i = 0; i < nfields; ++i) {
      m_intVariables.push_back(i);
    }
    nvariables = nfields;
  } else {
    nvariables = m_intVariables.size();
  }

  // Integrate in wave-space if using homogeneous1D
  if (m_HomogeneousType != eNotHomogeneous && m_homoInitialFwd) {
    for (i = 0; i < nfields; ++i) {
      m_fields[i]->HomogeneousFwdTrans(m_fields[i]->GetNpoints(),
                                       m_fields[i]->GetPhys(),
                                       m_fields[i]->UpdatePhys());
      m_fields[i]->SetWaveSpace(true);
      m_fields[i]->SetPhysState(false);
    }
  }

  // Set up wrapper to fields data storage.
  Array<OneD, Array<OneD, NekDouble>> fields(nvariables);

  // Order storage to list time-integrated fields first.
  for (i = 0; i < nvariables; ++i) {
    fields[i] = m_fields[m_intVariables[i]]->UpdatePhys();
    m_fields[m_intVariables[i]]->SetPhysState(false);
  }

  // Initialise time integration scheme
  m_intScheme->InitializeScheme(m_timestep, fields, m_time, m_ode);

  // Initialise filters
  for (auto &x : m_filters) {
    x.second->Initialise(m_fields, m_time);
  }

  LibUtilities::Timer timer;
  bool doCheckTime = false;
  int step = m_initialStep;
  int stepCounter = 0;
  int restartStep = -1;
  NekDouble intTime = 0.0;
  NekDouble cpuTime = 0.0;
  NekDouble cpuPrevious = 0.0;
  NekDouble elapsed = 0.0;
  NekDouble totFilterTime = 0.0;

  m_lastCheckTime = 0.0;

  Array<OneD, int> abortFlags(2, 0);
  std::string abortFile = "abort";
  if (m_session->DefinesSolverInfo("CheckAbortFile")) {
    abortFile = m_session->GetSolverInfo("CheckAbortFile");
  }

  NekDouble tmp_cflSafetyFactor = m_cflSafetyFactor;

  while ((step < m_steps || m_time < m_fintime - NekConstants::kNekZeroTol) &&
         abortFlags[1] == 0) {
    restartStep++;

    if (m_CFLGrowth > 1.0 && m_cflSafetyFactor < m_CFLEnd) {
      tmp_cflSafetyFactor =
          std::min(m_CFLEnd, m_CFLGrowth * tmp_cflSafetyFactor);
    }
    // Frozen preconditioner checks.
    if (!m_comm->IsParallelInTime()) {
      if (v_UpdateTimeStepCheck()) {
        m_cflSafetyFactor = tmp_cflSafetyFactor;

        if (m_cflSafetyFactor) {
          m_timestep = GetTimeStep(fields);
        }

        // Ensure that the final timestep finishes at the final
        // time, or at a prescribed IO_CheckTime.
        if (m_time + m_timestep > m_fintime && m_fintime > 0.0) {
          m_timestep = m_fintime - m_time;
        } else if (m_checktime &&
                   m_time + m_timestep - m_lastCheckTime >= m_checktime) {
          m_lastCheckTime += m_checktime;
          m_timestep = m_lastCheckTime - m_time;
          doCheckTime = true;
        }
      }
    }

    if (m_TimeIncrementFactor > 1.0) {
      NekDouble timeincrementFactor = m_TimeIncrementFactor;
      m_timestep *= timeincrementFactor;

      if (m_time + m_timestep > m_fintime && m_fintime > 0.0) {
        m_timestep = m_fintime - m_time;
      }
    }

    // Perform any solver-specific pre-integration steps
    timer.Start();
    if (v_PreIntegrate(step)) {
      break;
    }

    ASSERTL0(m_timestep > 0, "m_timestep < 0");

    fields = m_intScheme->TimeIntegrate(stepCounter, m_timestep);
    timer.Stop();

    m_time += m_timestep;
    elapsed = timer.TimePerTest(1);
    intTime += elapsed;
    cpuTime += elapsed;

    // Write out status information
    if (m_session->GetComm()->GetRank() == 0 && !((step + 1) % m_infosteps)) {
      std::cout << "Steps: " << std::setw(8) << std::left << step + 1 << " "
                << "Time: " << std::setw(12) << std::left << m_time;

      if (m_cflSafetyFactor) {
        std::cout << " Time-step: " << std::setw(12) << std::left << m_timestep;
      }

      std::stringstream ss;
      ss << cpuTime << "s";
      std::cout << " CPU Time: " << std::setw(8) << std::left << ss.str()
                << std::endl;
      cpuPrevious = cpuTime;
      cpuTime = 0.0;
    }

    // Transform data into coefficient space
    for (i = 0; i < nvariables; ++i) {
      // copy fields into ExpList::m_phys and assign the new
      // array to fields
      m_fields[m_intVariables[i]]->SetPhys(fields[i]);
      fields[i] = m_fields[m_intVariables[i]]->UpdatePhys();
      if (v_RequireFwdTrans()) {
        m_fields[m_intVariables[i]]->FwdTransLocalElmt(
            fields[i], m_fields[m_intVariables[i]]->UpdateCoeffs());
      }
      m_fields[m_intVariables[i]]->SetPhysState(false);
    }

    // Perform any solver-specific post-integration steps
    if (v_PostIntegrate(step)) {
      break;
    }

    // test for abort conditions (nan, or abort file)
    if (m_abortSteps && !((step + 1) % m_abortSteps)) {
      abortFlags[0] = 0;
      for (i = 0; i < nvariables; ++i) {
        if (Vmath::Nnan(fields[i].size(), fields[i], 1) > 0) {
          abortFlags[0] = 1;
        }
      }

      // rank zero looks for abort file and deltes it
      // if it exists. The communicates the abort
      if (m_session->GetComm()->GetRank() == 0) {
        if (std::filesystem::exists(abortFile)) {
          std::filesystem::remove(abortFile);
          abortFlags[1] = 1;
        }
      }

      m_session->GetComm()->AllReduce(abortFlags, LibUtilities::ReduceMax);

      ASSERTL0(!abortFlags[0], "NaN found during time integration.");
    }

    // Update filters
    for (auto &x : m_filters) {
      timer.Start();
      x.second->Update(m_fields, m_time);
      timer.Stop();
      elapsed = timer.TimePerTest(1);
      totFilterTime += elapsed;

      // Write out individual filter status information
      if (m_session->GetComm()->GetRank() == 0 &&
          !((step + 1) % m_filtersInfosteps) && !m_filters.empty() &&
          m_session->DefinesCmdLineArgument("verbose")) {
        std::stringstream s0;
        s0 << x.first << ":";
        std::stringstream s1;
        s1 << elapsed << "s";
        std::stringstream s2;
        s2 << elapsed / cpuPrevious * 100 << "%";
        std::cout << "CPU time for filter " << std::setw(25) << std::left
                  << s0.str() << std::setw(12) << std::left << s1.str()
                  << std::endl
                  << "\t Percentage of time integration:     " << std::setw(10)
                  << std::left << s2.str() << std::endl;
      }
    }

    // Write out overall filter status information
    if (m_session->GetComm()->GetRank() == 0 &&
        !((step + 1) % m_filtersInfosteps) && !m_filters.empty()) {
      std::stringstream ss;
      ss << totFilterTime << "s";
      std::cout << "Total filters CPU Time:\t\t\t     " << std::setw(10)
                << std::left << ss.str() << std::endl;
    }
    totFilterTime = 0.0;

    // Write out checkpoint files.
    if ((m_checksteps && !((step + 1) % m_checksteps)) || doCheckTime) {
      if (m_HomogeneousType != eNotHomogeneous) {
        // Transform to physical space for output.
        vector<bool> transformed(nfields, false);
        for (i = 0; i < nfields; i++) {
          if (m_fields[i]->GetWaveSpace()) {
            m_fields[i]->SetWaveSpace(false);
            m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(),
                                  m_fields[i]->UpdatePhys());
            transformed[i] = true;
          }
        }
        Checkpoint_Output(m_nchk);
        m_nchk++;

        // Transform back to wave-space after output.
        for (i = 0; i < nfields; i++) {
          if (transformed[i]) {
            m_fields[i]->SetWaveSpace(true);
            m_fields[i]->HomogeneousFwdTrans(m_fields[i]->GetTotPoints(),
                                             m_fields[i]->GetPhys(),
                                             m_fields[i]->UpdatePhys());
            m_fields[i]->SetPhysState(false);
          }
        }
      } else {
        Checkpoint_Output(m_nchk);
        m_nchk++;
      }
      doCheckTime = false;
    }

    // Step advance
    ++step;
    ++stepCounter;
  }

  // Print out summary statistics
  if (m_session->GetComm()->GetRank() == 0) {
    if (m_cflSafetyFactor > 0.0) {
      std::cout << "CFL safety factor : " << m_cflSafetyFactor << std::endl
                << "CFL time-step     : " << m_timestep << std::endl;
    }

    if (m_session->GetSolverInfo("Driver") != "SteadyState") {
      std::cout << "Time-integration  : " << intTime << "s" << std::endl;
    }
  }

  // If homogeneous, transform back into physical space if necessary.
  if (m_HomogeneousType != eNotHomogeneous) {
    for (i = 0; i < nfields; i++) {
      if (m_fields[i]->GetWaveSpace()) {
        m_fields[i]->SetWaveSpace(false);
        m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(),
                              m_fields[i]->UpdatePhys());
        m_fields[i]->SetPhysState(true);
      }
    }
  } else {
    for (i = 0; i < nvariables; ++i) {
      m_fields[m_intVariables[i]]->SetPhys(fields[i]);
      m_fields[m_intVariables[i]]->SetPhysState(true);
    }
  }

  // Finalise filters
  for (auto &x : m_filters) {
    x.second->Finalise(m_fields, m_time);
  }
}

/**
 * @brief Compute the advection terms for the right-hand side
 */
void SOLSystem::DoAdvection(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time,
    const Array<OneD, const Array<OneD, NekDouble>> &pFwd,
    const Array<OneD, const Array<OneD, NekDouble>> &pBwd) {
  // Only fields up to and including the energy need to be advected
  int num_fields_to_advect = m_field_to_index.get_idx("E") + 1;

  Array<OneD, Array<OneD, NekDouble>> advVel(m_spacedim);
  m_advObject->Advect(num_fields_to_advect, m_fields, advVel, inarray, outarray,
                      time, pFwd, pBwd);
}

/**
 * @brief Add the diffusions terms to the right-hand side
 */
void SOLSystem::DoDiffusion(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray,
    const Array<OneD, const Array<OneD, NekDouble>> &pFwd,
    const Array<OneD, const Array<OneD, NekDouble>> &pBwd) {
  boost::ignore_unused(inarray, outarray, pFwd, pBwd);
  // Do nothing for now
}

/**
 * @brief Return the flux vector for the compressible Euler equations.
 *
 * @param physfield   Fields.
 * @param flux        Resulting flux.
 */
void SOLSystem::GetFluxVector(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    TensorOfArray3D<NekDouble> &flux) {
  // Energy is the last field of relevance, regardless of mesh dimension
  const auto E_idx = m_field_to_index.get_idx("E");
  const auto nVariables = E_idx + 1;
  const auto nPts = physfield[0].size();

  // Temporary space for 2 velocity fields; second one is ignored in 1D
  constexpr unsigned short num_all_flds = 4;
  constexpr unsigned short num_vel_flds = 2;

  for (std::size_t p = 0; p < nPts; ++p) {
    // Create local storage
    std::array<NekDouble, num_all_flds> all_phys;
    std::array<NekDouble, num_vel_flds> vel_phys;

    // Copy phys vals for this point
    for (std::size_t f = 0; f < nVariables; ++f) {
      all_phys[f] = physfield[f][p];
    }

    // 1 / rho
    NekDouble oneOrho = 1.0 / all_phys[0];

    for (std::size_t dim = 0; dim < m_spacedim; ++dim) {
      // Add momentum densities to flux vector
      flux[0][dim][p] = all_phys[dim + 1];
      // Compute velocities from momentum densities
      vel_phys[dim] = all_phys[dim + 1] * oneOrho;
    }

    NekDouble pressure = m_varConv->GetPressure(all_phys.data());
    NekDouble ePlusP = all_phys[E_idx] + pressure;
    for (auto dim = 0; dim < m_spacedim; ++dim) {
      // Flux vector for the velocity fields
      for (auto vdim = 0; vdim < m_spacedim; ++vdim) {
        flux[dim + 1][vdim][p] = vel_phys[vdim] * all_phys[dim + 1];
      }

      // Add pressure to appropriate field
      flux[dim + 1][dim][p] += pressure;

      // Energy flux
      flux[m_spacedim + 1][dim][p] = ePlusP * vel_phys[dim];
    }
  }
}

/**
 * @brief Calculate the maximum timestep on each element
 *        subject to CFL restrictions.
 */
void SOLSystem::GetElmtTimeStep(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, NekDouble> &tstep) {
  boost::ignore_unused(inarray);

  int nElements = m_fields[0]->GetExpSize();

  // Change value of m_timestep (in case it is set to zero)
  NekDouble tmp = m_timestep;
  m_timestep = 1.0;

  Array<OneD, NekDouble> cfl(nElements);
  cfl = GetElmtCFLVals();

  // Factors to compute the time-step limit
  NekDouble alpha = MaxTimeStepEstimator();

  // Loop over elements to compute the time-step limit for each element
  for (int n = 0; n < nElements; ++n) {
    tstep[n] = m_cflSafetyFactor * alpha / cfl[n];
  }

  // Restore value of m_timestep
  m_timestep = tmp;
}

/**
 * @brief Calculate the maximum timestep subject to CFL restrictions.
 */
NekDouble SOLSystem::v_GetTimeStep(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray) {
  int nElements = m_fields[0]->GetExpSize();
  Array<OneD, NekDouble> tstep(nElements, 0.0);

  GetElmtTimeStep(inarray, tstep);

  // Get the minimum time-step limit and return the time-step
  NekDouble TimeStep = Vmath::Vmin(nElements, tstep, 1);
  m_comm->AllReduce(TimeStep, LibUtilities::ReduceMin);

  return TimeStep;
}

/**
 * @brief Compute the advection velocity in the standard space
 * for each element of the expansion.
 */
Array<OneD, NekDouble>
SOLSystem::v_GetMaxStdVelocity(const NekDouble SpeedSoundFactor) {
  int nTotQuadPoints = GetTotPoints();
  int n_element = m_fields[0]->GetExpSize();
  int expdim = m_fields[0]->GetGraph()->GetMeshDimension();
  int nfields = m_fields.size();
  int offset;
  Array<OneD, NekDouble> tmp;

  Array<OneD, Array<OneD, NekDouble>> physfields(nfields);
  for (int i = 0; i < nfields; ++i) {
    physfields[i] = m_fields[i]->GetPhys();
  }

  Array<OneD, NekDouble> stdV(n_element, 0.0);

  // Getting the velocity vector on the 2D normal space
  Array<OneD, Array<OneD, NekDouble>> velocity(m_spacedim);
  Array<OneD, Array<OneD, NekDouble>> stdVelocity(m_spacedim);
  Array<OneD, Array<OneD, NekDouble>> stdSoundSpeed(m_spacedim);
  Array<OneD, NekDouble> soundspeed(nTotQuadPoints);
  LibUtilities::PointsKeyVector ptsKeys;

  for (int i = 0; i < m_spacedim; ++i) {
    velocity[i] = Array<OneD, NekDouble>(nTotQuadPoints);
    stdVelocity[i] = Array<OneD, NekDouble>(nTotQuadPoints, 0.0);
    stdSoundSpeed[i] = Array<OneD, NekDouble>(nTotQuadPoints, 0.0);
  }

  m_varConv->GetVelocityVector(physfields, velocity);
  m_varConv->GetSoundSpeed(physfields, soundspeed);

  for (int el = 0; el < n_element; ++el) {
    ptsKeys = m_fields[0]->GetExp(el)->GetPointsKeys();
    offset = m_fields[0]->GetPhys_Offset(el);
    int nq = m_fields[0]->GetExp(el)->GetTotPoints();

    const SpatialDomains::GeomFactorsSharedPtr metricInfo =
        m_fields[0]->GetExp(el)->GetGeom()->GetMetricInfo();
    const Array<TwoD, const NekDouble> &gmat =
        m_fields[0]->GetExp(el)->GetGeom()->GetMetricInfo()->GetDerivFactors(
            ptsKeys);

    // Convert to standard element
    //    consider soundspeed in all directions
    //    (this might overestimate the cfl)
    if (metricInfo->GetGtype() == SpatialDomains::eDeformed) {
      // d xi/ dx = gmat = 1/J * d x/d xi
      for (int i = 0; i < expdim; ++i) {
        Vmath::Vmul(nq, gmat[i], 1, velocity[0] + offset, 1,
                    tmp = stdVelocity[i] + offset, 1);
        Vmath::Vmul(nq, gmat[i], 1, soundspeed + offset, 1,
                    tmp = stdSoundSpeed[i] + offset, 1);
        for (int j = 1; j < expdim; ++j) {
          Vmath::Vvtvp(nq, gmat[expdim * j + i], 1, velocity[j] + offset, 1,
                       stdVelocity[i] + offset, 1,
                       tmp = stdVelocity[i] + offset, 1);
          Vmath::Vvtvp(nq, gmat[expdim * j + i], 1, soundspeed + offset, 1,
                       stdSoundSpeed[i] + offset, 1,
                       tmp = stdSoundSpeed[i] + offset, 1);
        }
      }
    } else {
      for (int i = 0; i < expdim; ++i) {
        Vmath::Smul(nq, gmat[i][0], velocity[0] + offset, 1,
                    tmp = stdVelocity[i] + offset, 1);
        Vmath::Smul(nq, gmat[i][0], soundspeed + offset, 1,
                    tmp = stdSoundSpeed[i] + offset, 1);
        for (int j = 1; j < expdim; ++j) {
          Vmath::Svtvp(nq, gmat[expdim * j + i][0], velocity[j] + offset, 1,
                       stdVelocity[i] + offset, 1,
                       tmp = stdVelocity[i] + offset, 1);
          Vmath::Svtvp(nq, gmat[expdim * j + i][0], soundspeed + offset, 1,
                       stdSoundSpeed[i] + offset, 1,
                       tmp = stdSoundSpeed[i] + offset, 1);
        }
      }
    }

    NekDouble vel;
    for (int i = 0; i < nq; ++i) {
      NekDouble pntVelocity = 0.0;
      for (int j = 0; j < expdim; ++j) {
        // Add sound speed
        vel = std::abs(stdVelocity[j][offset + i]) +
              SpeedSoundFactor * std::abs(stdSoundSpeed[j][offset + i]);
        pntVelocity += vel * vel;
      }
      pntVelocity = sqrt(pntVelocity);
      if (pntVelocity > stdV[el]) {
        stdV[el] = pntVelocity;
      }
    }
  }

  return stdV;
}

/**
 *
 */
void SOLSystem::v_GetPressure(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &pressure) {
  m_varConv->GetPressure(physfield, pressure);
}

/**
 *
 */
void SOLSystem::v_GetDensity(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &density) {
  density = physfield[0];
}

/**
 *
 */
void SOLSystem::v_GetVelocity(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, NekDouble>> &velocity) {
  m_varConv->GetVelocityVector(physfield, velocity);
}

} // namespace Nektar
