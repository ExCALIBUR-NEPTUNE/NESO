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
// Description: Compressible flow system base class with auxiliary functions
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/core/ignore_unused.hpp>

#include "SOLSystem.h"

using namespace std;

namespace Nektar {
string SOLSystem::className =
    SolverUtils::GetEquationSystemFactory().RegisterCreatorFunction(
        "SOL", SOLSystem::create, "SOL equations in conservative variables.");

SOLSystem::SOLSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
                     const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph), AdvectionSystem(pSession, pGraph) {}

/**
 * @brief Initialization object for SOLSystem class.
 */
void SOLSystem::v_InitObject(bool DeclareField) {
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

  string advName, riemName;
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

  m_bndEvaluateTime = time;

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

  // Add diffusion terms
  DoDiffusion(inarray, outarray, Fwd, Bwd);

  // Add forcing terms
  for (auto &x : m_forcing) {
    x->Apply(m_fields, inarray, outarray, time);
  }
}

/**
 * @brief Compute the projection and call the method for imposing the
 * boundary conditions in case of discontinuous projection.
 */
void SOLSystem::DoOdeProjection(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {
  // Just copy over array
  int nvariables = inarray.size();
  int npoints = GetNpoints();

  for (int i = 0; i < nvariables; ++i) {
    Vmath::Vcopy(npoints, inarray[i], 1, outarray[i], 1);
    SetBoundaryConditions(outarray, time);
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
  int nvariables = inarray.size();
  Array<OneD, Array<OneD, NekDouble>> advVel(m_spacedim);

  m_advObject->Advect(nvariables, m_fields, advVel, inarray, outarray, time,
                      pFwd, pBwd);
}

/**
 * @brief Add the diffusions terms to the right-hand side
 */
void SOLSystem::DoDiffusion(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray,
    const Array<OneD, const Array<OneD, NekDouble>> &pFwd,
    const Array<OneD, const Array<OneD, NekDouble>> &pBwd) {
  // Do nothing for now
}

void SOLSystem::SetBoundaryConditions(
    Array<OneD, Array<OneD, NekDouble>> &physarray, NekDouble time) {
  int nTracePts = GetTraceTotPoints();
  int nvariables = physarray.size();

  Array<OneD, Array<OneD, NekDouble>> Fwd(nvariables);
  for (int i = 0; i < nvariables; ++i) {
    Fwd[i] = Array<OneD, NekDouble>(nTracePts);
    m_fields[i]->ExtractTracePhys(physarray[i], Fwd[i]);
  }

  // if (m_bndConds.size())
  // {
  //     // Loop over user-defined boundary conditions
  //     for (auto &x : m_bndConds)
  //     {
  //         x->Apply(Fwd, physarray, time);
  //     }
  // }
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
  auto nVariables = physfield.size();
  auto nPts = physfield[0].size();

  constexpr unsigned short maxVel = 2;
  constexpr unsigned short maxFld = 4;

  // hardcoding done for performance reasons
  ASSERTL1(nVariables <= maxFld, "GetFluxVector, hard coded max fields");

  for (size_t p = 0; p < nPts; ++p) {
    // local storage
    std::array<NekDouble, maxFld> fieldTmp;
    std::array<NekDouble, maxVel> velocity;

    // rearrange and load data
    for (size_t f = 0; f < nVariables; ++f) {
      fieldTmp[f] = physfield[f][p]; // load
    }

    // 1 / rho
    NekDouble oneOrho = 1.0 / fieldTmp[0];

    for (size_t d = 0; d < m_spacedim; ++d) {
      // Flux vector for the rho equation
      flux[0][d][p] = fieldTmp[d + 1]; // store
      // compute velocities from momentum densities
      velocity[d] = fieldTmp[d + 1] * oneOrho;
    }

    NekDouble pressure = m_varConv->GetPressure(fieldTmp.data());
    NekDouble ePlusP = fieldTmp[m_spacedim + 1] + pressure;
    for (size_t f = 0; f < m_spacedim; ++f) {
      // Flux vector for the velocity fields
      for (size_t d = 0; d < m_spacedim; ++d) {
        flux[f + 1][d][p] = velocity[d] * fieldTmp[f + 1]; // store
      }

      // Add pressure to appropriate field
      flux[f + 1][f][p] += pressure;

      // Flux vector for energy
      flux[m_spacedim + 1][f][p] = ePlusP * velocity[f]; // store
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

  NekDouble tmp = m_timestep;
  m_timestep = TimeStep;

  Array<OneD, NekDouble> cflNonAcoustic(nElements, 0.0);
  cflNonAcoustic = GetElmtCFLVals(false);

  // Get the minimum time-step limit and return the time-step
  NekDouble MaxcflNonAcoustic = Vmath::Vmax(nElements, cflNonAcoustic, 1);
  m_comm->AllReduce(MaxcflNonAcoustic, LibUtilities::ReduceMax);

  m_cflNonAcoustic = MaxcflNonAcoustic;
  m_timestep = tmp;

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
void SOLSystem::GetPressure(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &pressure) {
  m_varConv->GetPressure(physfield, pressure);
}

/**
 *
 */
void SOLSystem::GetDensity(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, NekDouble> &density) {
  density = physfield[0];
}

/**
 *
 */
void SOLSystem::GetVelocity(
    const Array<OneD, const Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, NekDouble>> &velocity) {
  m_varConv->GetVelocityVector(physfield, velocity);
}

void SOLSystem::v_SteadyStateResidual(int step, Array<OneD, NekDouble> &L2) {
  boost::ignore_unused(step);
  const int nPoints = GetTotPoints();
  const int nFields = m_fields.size();
  Array<OneD, Array<OneD, NekDouble>> rhs(nFields);
  Array<OneD, Array<OneD, NekDouble>> inarray(nFields);
  for (int i = 0; i < nFields; ++i) {
    rhs[i] = Array<OneD, NekDouble>(nPoints, 0.0);
    inarray[i] = m_fields[i]->UpdatePhys();
  }

  DoOdeRhs(inarray, rhs, m_time);

  // Holds L2 errors.
  Array<OneD, NekDouble> tmp;
  Array<OneD, NekDouble> RHSL2(nFields);
  Array<OneD, NekDouble> residual(nFields);

  for (int i = 0; i < nFields; ++i) {
    tmp = rhs[i];

    Vmath::Vmul(nPoints, tmp, 1, tmp, 1, tmp, 1);
    residual[i] = Vmath::Vsum(nPoints, tmp, 1);
  }

  m_comm->AllReduce(residual, LibUtilities::ReduceSum);

  NekDouble onPoints = 1.0 / NekDouble(nPoints);
  for (int i = 0; i < nFields; ++i) {
    L2[i] = sqrt(residual[i] * onPoints);
  }
}

/**
 * @brief Compute an estimate of minimum h/p for each element of the expansion.
 */
Array<OneD, NekDouble> SOLSystem::GetElmtMinHP(void) {
  int nElements = m_fields[0]->GetExpSize();
  Array<OneD, NekDouble> hOverP(nElements, 1.0);

  // Determine h/p scaling
  Array<OneD, int> pOrderElmt = m_fields[0]->EvalBasisNumModesMaxPerExp();
  for (int e = 0; e < nElements; e++) {
    NekDouble h = 1.0e+10;

    LocalRegions::Expansion1DSharedPtr exp1D;
    exp1D = m_fields[0]->GetExp(e)->as<LocalRegions::Expansion1D>();
    h = min(h, exp1D->GetGeom1D()->GetVertex(0)->dist(
                   *(exp1D->GetGeom1D()->GetVertex(1))));

    // Determine h/p scaling
    hOverP[e] = h / max(pOrderElmt[e] - 1, 1);
  }

  return hOverP;
}

} // namespace Nektar
