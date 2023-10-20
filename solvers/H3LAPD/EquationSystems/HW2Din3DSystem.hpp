///////////////////////////////////////////////////////////////////////////////
//
// File HW2Din3DSystem.hpp
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
// Description: Header for a reduced version of the Hermes-3 LAPD equation
// system
//
///////////////////////////////////////////////////////////////////////////////

#ifndef HW2Din3D_H
#define HW2Din3D_H

#include "nektar_interface/utilities.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/EquationSystem.h>
#include <SolverUtils/Forcing/Forcing.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>
#include <solvers/solver_callback_handler.hpp>

#include "../Diagnostics/GrowthRatesRecorder.hpp"
#include "../Diagnostics/MassRecorder.hpp"
#include "DriftReducedSystem.hpp"

namespace Nektar {

class HW2Din3DSystem : virtual public DriftReducedSystem {
public:
  friend class MemoryManager<HW2Din3DSystem>;

  /// Name of class.
  static std::string className;

  /// Creates an instance of this class.
  static SolverUtils::EquationSystemSharedPtr
  create(const LibUtilities::SessionReaderSharedPtr &pSession,
         const SpatialDomains::MeshGraphSharedPtr &pGraph) {
    SolverUtils::EquationSystemSharedPtr p =
        MemoryManager<HW2Din3DSystem>::AllocateSharedPtr(pSession, pGraph);
    p->InitObject();
    return p;
  }

  //---------------------------------------------------------------------------
  // Diagnostics

  // Flags to toggle recorders
  bool m_diag_growth_rates_recording_enabled;
  bool m_diag_mass_recording_enabled;

  // Object that allows optional recording of total fluid, particle masses
  std::shared_ptr<MassRecorder<MultiRegions::DisContField>>
      m_diag_mass_recorder;

  // Object that allows optional recording of energy and enstrophy growth rates
  std::shared_ptr<GrowthRatesRecorder<MultiRegions::DisContField>>
      m_diag_growth_rates_recorder;

  // Callback handler to call user defined callbacks.
  SolverCallbackHandler<HW2Din3DSystem> m_solver_callback_handler;

protected:
  HW2Din3DSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
                 const SpatialDomains::MeshGraphSharedPtr &pGraph);

  virtual void CalcEAndAdvVels(
      const Array<OneD, const Array<OneD, NekDouble>> &inarray) override;

  void ExplicitTimeInt(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble time) override;

  void GetPhiSolveRHS(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                      Array<OneD, NekDouble> &rhs) override;

  void LoadParams() override;

  virtual void v_InitObject(bool DeclareField) override;

  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;

private:
  NekDouble m_alpha;
  NekDouble m_kappa;

  void UpdateEnergy();
};

} // namespace Nektar
#endif
