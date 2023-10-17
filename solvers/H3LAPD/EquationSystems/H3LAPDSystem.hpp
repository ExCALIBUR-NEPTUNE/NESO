///////////////////////////////////////////////////////////////////////////////
//
// File H3LAPDSystem.hpp
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
// Description: Header for the Hermes-3 LAPD equation system
//
///////////////////////////////////////////////////////////////////////////////

#ifndef H3LAPDSYSTEM_H
#define H3LAPDSYSTEM_H

#include "../Diagnostics/GrowthRatesRecorder.hpp"
#include "../ParticleSystems/neutral_particles.hpp"

#include "nektar_interface/utilities.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/EquationSystem.h>
#include <SolverUtils/Forcing/Forcing.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

#include <solvers/solver_callback_handler.hpp>
namespace Nektar {

class H3LAPDSystem : virtual public SolverUtils::AdvectionSystem {
public:
  friend class MemoryManager<H3LAPDSystem>;

  /// Name of class.
  static std::string className;

  /// Creates an instance of this class.
  static SolverUtils::EquationSystemSharedPtr
  create(const LibUtilities::SessionReaderSharedPtr &pSession,
         const SpatialDomains::MeshGraphSharedPtr &pGraph) {
    SolverUtils::EquationSystemSharedPtr p =
        MemoryManager<H3LAPDSystem>::AllocateSharedPtr(pSession, pGraph);
    p->InitObject();
    return p;
  }

  /// Free particle system memory on destruction.
  virtual ~H3LAPDSystem() { m_particle_sys->free(); }

protected:
  /// Protected constructor. Since we use a factory pattern, objects should be
  /// constructed via the SolverUtils::EquationSystem factory.
  H3LAPDSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
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

  void
  AddCollisionTerms(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray);

  void AddDensitySource(Array<OneD, Array<OneD, NekDouble>> &outarray);
  void AddEParTerms(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray);
  void AddGradPTerms(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                     Array<OneD, Array<OneD, NekDouble>> &outarray);

  void AddParticleSources(std::vector<std::string> target_fields,
                          Array<OneD, Array<OneD, NekDouble>> &outarray);
  void CalcCollisionFreqs(const Array<OneD, NekDouble> &ne,
                          Array<OneD, NekDouble> &coeffs);
  void CalcCoulombLogarithm(const Array<OneD, NekDouble> &ne,
                            Array<OneD, NekDouble> &LogLambda);
  void
  CalcEAndAdvVels(const Array<OneD, const Array<OneD, NekDouble>> &inarray);
  void CalcAdvNormalVels();
  void DoOdeProjection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble time);

  virtual void
  ExplicitTimeInt(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                  Array<OneD, Array<OneD, NekDouble>> &outarray,
                  const NekDouble time);
  void GetFluxVectorDiff(
      const Array<OneD, Array<OneD, NekDouble>> &inarray,
      const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &qfield,
      Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscousTensor);

  void
  GetFluxVectorElec(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);
  void
  GetFluxVectorIons(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);
  void
  GetFluxVectorVort(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  void GetFluxVectorPD(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                       Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  virtual void
  GetPhiSolveRHS(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                 Array<OneD, NekDouble> &rhs);

  Array<OneD, NekDouble> &
  GetVnAdv(Array<OneD, NekDouble> &traceVn,
           const Array<OneD, Array<OneD, NekDouble>> &vAdv);

  Array<OneD, NekDouble> &GetVnAdvElec();
  Array<OneD, NekDouble> &GetVnAdvIons();
  Array<OneD, NekDouble> &GetVnAdvVort();
  Array<OneD, NekDouble> &GetVnAdvPD();

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
  // Charge unit
  NekDouble m_charge_e;
  // Ion mass;
  NekDouble m_md;
  // Electron mass;
  NekDouble m_me;
  // Factor used to set the density floor (n_floor = m_n_floor_fac * m_nRef)
  NekDouble m_n_floor_fac;
  // Reference number density
  NekDouble m_nRef;
  // Riemann solver type (used for all advection terms)
  std::string m_RiemSolvType;
  // Ion temperature in eV
  NekDouble m_Td;
  // Electron temperature in eV
  NekDouble m_Te;
  //---------------------------------------------------------------------------
  // Coefficient factors for Helmsolve
  NekDouble m_d00;
  NekDouble m_d11;
  NekDouble m_d22;
  //---------------------------------------------------------------------------
  // Factors used in collision coeff calculation
  // Density-independent part of the Coulomb logarithm; read from config
  NekDouble m_coulomb_log_const;
  // Pre-factor used when calculating collision frequencies; read from config
  NekDouble m_nu_ei_const;
  // Factor to convert densities (back) to SI; used in Coulomb logarithm calc
  NekDouble m_n_to_SI;
  //---------------------------------------------------------------------------
  // Advection objects
  SolverUtils::AdvectionSharedPtr m_advElec;
  SolverUtils::AdvectionSharedPtr m_advIons;
  SolverUtils::AdvectionSharedPtr m_advVort;
  SolverUtils::AdvectionSharedPtr m_advPD;
  // Storage for Electric field
  Array<OneD, Array<OneD, NekDouble>> m_E;
  // Riemann solver objects
  SolverUtils::RiemannSolverSharedPtr m_riemannSolverElec;
  SolverUtils::RiemannSolverSharedPtr m_riemannSolverIons;
  SolverUtils::RiemannSolverSharedPtr m_riemannSolverPD;
  SolverUtils::RiemannSolverSharedPtr m_riemannSolverVort;
  // Storage for advection velocities dotted with element face normals
  Array<OneD, NekDouble> m_traceVnElec;
  Array<OneD, NekDouble> m_traceVnIons;
  Array<OneD, NekDouble> m_traceVnPD;
  Array<OneD, NekDouble> m_traceVnVort;
  // Storage for electron, ion advection velocities
  Array<OneD, Array<OneD, NekDouble>> m_vAdvElec;
  Array<OneD, Array<OneD, NekDouble>> m_vAdvIons;
  /*Storage for difference between elec, ion parallel velocities. Has size ndim
   * so that it can be used in advection operation; non-parallel values are 0
   */
  Array<OneD, Array<OneD, NekDouble>> m_vAdvDiffPar;
  // Storage for ExB drift velocity
  Array<OneD, Array<OneD, NekDouble>> m_vExB;
  // Storage for electron, ion parallel velocities
  Array<OneD, NekDouble> m_vParIons;
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
