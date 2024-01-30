#ifndef H3LAPD_HW_SYSTEM_H
#define H3LAPD_HW_SYSTEM_H

#include "nektar_interface/utilities.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/EquationSystem.h>
#include <solvers/solver_callback_handler.hpp>

#include "../Diagnostics/GrowthRatesRecorder.hpp"
#include "../Diagnostics/MassRecorder.hpp"
#include "DriftReducedSystem.hpp"

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Solvers::H3LAPD {

/**
 * @brief Base class for 2D-in-3D and true 3D Hasegawa-Wakatani equation
 * systems.
 */
class HWSystem : virtual public DriftReducedSystem {
public:
  friend class MemoryManager<HWSystem>;

  /// Object that allows optional recording of energy and enstrophy growth rates
  std::shared_ptr<GrowthRatesRecorder<MR::DisContField>>
      m_diag_growth_rates_recorder;
  /// Object that allows optional recording of total fluid, particle masses
  std::shared_ptr<MassRecorder<MR::DisContField>> m_diag_mass_recorder;
  /// Callback handler to call user defined callbacks.
  SolverCallbackHandler<HWSystem> m_solver_callback_handler;

protected:
  HWSystem(const LU::SessionReaderSharedPtr &session,
           const SD::MeshGraphSharedPtr &graph);

  virtual void calc_E_and_adv_vels(
      const Array<OneD, const Array<OneD, NekDouble>> &inarray) override final;

  void
  get_phi_solve_rhs(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, NekDouble> &rhs) override final;

  void v_GenerateSummary(SU::SummaryList &s);
  virtual void v_InitObject(bool DeclareField) override;

  virtual bool v_PostIntegrate(int step) override final;
  virtual bool v_PreIntegrate(int step) override final;

  /// Bool to enable/disable growth rate recordings
  bool m_diag_growth_rates_recording_enabled;
  /// Bool to enable/disable mass recordings
  bool m_diag_mass_recording_enabled;
  /// Hasegawa-Wakatani α
  NekDouble m_alpha;
  /// Hasegawa-Wakatani κ
  NekDouble m_kappa;
};

} // namespace NESO::Solvers::H3LAPD

#endif