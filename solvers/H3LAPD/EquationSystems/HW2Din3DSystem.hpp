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

#ifndef H3LAPD_HW2DIN3D_SYSTEM_H
#define H3LAPD_HW2DIN3D_SYSTEM_H

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

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::H3LAPD {

class HW2Din3DSystem : virtual public DriftReducedSystem {
public:
  friend class MemoryManager<HW2Din3DSystem>;

  /**
   * @brief Creates an instance of this class.
   */
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<HW2Din3DSystem>::AllocateSharedPtr(session, graph);
    p->InitObject();
    return p;
  }

  /// Name of class
  static std::string class_name;
  /// Object that allows optional recording of energy and enstrophy growth rates
  std::shared_ptr<GrowthRatesRecorder<MR::DisContField>>
      m_diag_growth_rates_recorder;
  /// Object that allows optional recording of total fluid, particle masses
  std::shared_ptr<MassRecorder<MR::DisContField>> m_diag_mass_recorder;
  /// Callback handler to call user defined callbacks.
  SolverCallbackHandler<HW2Din3DSystem> m_solver_callback_handler;

protected:
  HW2Din3DSystem(const LU::SessionReaderSharedPtr &session,
                 const SD::MeshGraphSharedPtr &graph);

  virtual void calc_E_and_adv_vels(
      const Array<OneD, const Array<OneD, NekDouble>> &inarray) override;

  void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray,
                    const NekDouble time) override;

  void
  get_phi_solve_rhs(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, NekDouble> &rhs) override;

  void load_params() override;

  virtual void v_InitObject(bool DeclareField) override;

  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;

private:
  /// Hasegawa-Wakatani α
  NekDouble m_alpha;
  /// Bool to enable/disable growth rate recordings
  bool m_diag_growth_rates_recording_enabled;
  /// Bool to enable/disable mass recordings
  bool m_diag_mass_recording_enabled;
  /// Hasegawa-Wakatani κ
  NekDouble m_kappa;
};

} // namespace NESO::Solvers::H3LAPD
#endif // H3LAPD_HW2DIN3D_SYSTEM_H