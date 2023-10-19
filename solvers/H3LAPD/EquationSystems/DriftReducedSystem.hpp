///////////////////////////////////////////////////////////////////////////////
//
// File DriftReducedSystem.hpp
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
// Description: Base class for drift reduced systems.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DRIFTREDUCEDSYSTEM_H
#define DRIFTREDUCEDSYSTEM_H

#include "../ParticleSystems/neutral_particles.hpp"

#include "nektar_interface/utilities.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/EquationSystem.h>
#include <SolverUtils/Forcing/Forcing.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

#include <solvers/solver_callback_handler.hpp>
namespace Nektar {

class DriftReducedSystem : virtual public SolverUtils::AdvectionSystem {
public:
  friend class MemoryManager<DriftReducedSystem>;

  /// Name of class.
  static std::string className;

  /// Free particle system memory on destruction.
  virtual ~DriftReducedSystem() { m_particle_sys->free(); }

protected:
  DriftReducedSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
                     const SpatialDomains::MeshGraphSharedPtr &pGraph);

  // Field name => index mapper
  NESO::NektarFieldIndexMap m_field_to_index;

  // List of field names required by the solver
  std::vector<std::string> m_required_flds;
  // Names of fields that will be time integrated
  std::vector<std::string> m_int_fld_names;

  void
  AddAdvTerms(std::vector<std::string> field_names,
              const SolverUtils::AdvectionSharedPtr advObj,
              const Array<OneD, Array<OneD, NekDouble>> &vAdv,
              const Array<OneD, const Array<OneD, NekDouble>> &inarray,
              Array<OneD, Array<OneD, NekDouble>> &outarray,
              const NekDouble time,
              std::vector<std::string> eqn_labels = std::vector<std::string>());

  void AddDensitySource(Array<OneD, Array<OneD, NekDouble>> &outarray);

  void AddParticleSources(std::vector<std::string> target_fields,
                          Array<OneD, Array<OneD, NekDouble>> &outarray);
  virtual void
  CalcEAndAdvVels(const Array<OneD, const Array<OneD, NekDouble>> &inarray);
  void DoOdeProjection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble time);

  virtual void
  ExplicitTimeInt(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                  Array<OneD, Array<OneD, NekDouble>> &outarray,
                  const NekDouble time) = 0;
  void GetFluxVectorDiff(
      const Array<OneD, Array<OneD, NekDouble>> &inarray,
      const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &qfield,
      Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscousTensor);

  void GetFluxVector(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                     const Array<OneD, Array<OneD, NekDouble>> &vAdv,
                     Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);
  void
  GetFluxVectorElec(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);
  void
  GetFluxVectorVort(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  virtual void
  GetPhiSolveRHS(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                 Array<OneD, NekDouble> &rhs) = 0;

  Array<OneD, NekDouble> &
  GetVnAdv(Array<OneD, NekDouble> &traceVn,
           const Array<OneD, Array<OneD, NekDouble>> &vAdv);

  Array<OneD, NekDouble> &GetVnAdvElec();
  Array<OneD, NekDouble> &GetVnAdvVort();

  virtual void LoadParams();

  void SolvePhi(const Array<OneD, const Array<OneD, NekDouble>> &inarray);

  void ValidateFieldList();

  virtual void v_InitObject(bool DeclareField) override;
  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;

  void ZeroOutArray(Array<OneD, Array<OneD, NekDouble>> &outarray);

  // Advection type
  std::string m_advType;
  // Magnetic field vector
  std::vector<NekDouble> m_B;
  // Magnitude of the magnetic field
  NekDouble m_Bmag;
  // Normalised magnetic field vector
  std::vector<NekDouble> m_b_unit;
  // Factor used to set the density floor (n_floor = m_n_floor_fac * m_nRef)
  NekDouble m_n_floor_fac;
  // Reference number density
  NekDouble m_nRef;
  // Riemann solver type (used for all advection terms)
  std::string m_riemann_solver_type;
  //---------------------------------------------------------------------------
  // Coefficient factors for Helmsolve
  NekDouble m_d00;
  NekDouble m_d11;
  NekDouble m_d22;
  //---------------------------------------------------------------------------
  // Advection objects
  SolverUtils::AdvectionSharedPtr m_advElec;
  SolverUtils::AdvectionSharedPtr m_advVort;
  // Storage for Electric field
  Array<OneD, Array<OneD, NekDouble>> m_E;
  // Riemann solver objects for electron and vorticity advection
  SolverUtils::RiemannSolverSharedPtr m_riemannSolverElec;
  SolverUtils::RiemannSolverSharedPtr m_riemannSolverVort;
  // Storage for advection velocities dotted with element face normals
  Array<OneD, NekDouble> m_traceVnElec;
  Array<OneD, NekDouble> m_traceVnVort;
  // Storage for electron advection velocities
  Array<OneD, Array<OneD, NekDouble>> m_vAdvElec;
  // Storage for ExB drift velocity
  Array<OneD, Array<OneD, NekDouble>> m_vExB;
  // Storage for electron parallel velocities
  Array<OneD, NekDouble> m_vParElec;

  //---------------------------------------------------------------------------
  // Particles system and associated parameters
  std::shared_ptr<NeutralParticleSystem> m_particle_sys;

  // Number of particle timesteps per fluid timestep.
  int m_num_part_substeps;
  // Number of time steps between particle trajectory step writes.
  int m_num_write_particle_steps;
  // Particle timestep size.
  double m_part_timestep;

  /*
  Source fields cast to DisContFieldSharedPtr, indexed by name, for use in
  particle evaluation/projection methods
  */
  std::map<std::string, MultiRegions::DisContFieldSharedPtr> m_discont_fields;

  //---------------------------------------------------------------------------
  // Debugging
  void PrintArrVals(const Array<OneD, NekDouble> &arr, int num, int stride = 1,
                    std::string label = "", bool all_tasks = false);
  void PrintArrSize(const Array<OneD, NekDouble> &arr, std::string label = "",
                    bool all_tasks = false);
};

} // namespace Nektar
#endif
