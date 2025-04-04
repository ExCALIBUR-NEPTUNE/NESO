#ifndef __NESOSOLVERS_DRIFTREDUCED_HWSYSTEM_HPP__
#define __NESOSOLVERS_DRIFTREDUCED_HWSYSTEM_HPP__

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/EquationSystem.h>
#include <nektar_interface/utilities.hpp>
#include <solvers/solver_callback_handler.hpp>

#include "../Diagnostics/GrowthRatesRecorder.hpp"
#include "../Diagnostics/MassRecorder.hpp"
#include "DriftReducedSystem.hpp"

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Solvers::DriftReduced {

/**
 * @brief Base class for Hasegawa-Wakatani equation systems.
 */
class HWSystem : public DriftReducedSystem {
public:
  friend class Nektar::MemoryManager<HWSystem>;

  /// Object that allows optional recording of energy and enstrophy growth rates
  std::shared_ptr<GrowthRatesRecorder<MR::DisContField>>
      diag_growth_rates_recorder;
  /// Object that allows optional recording of total fluid, particle masses
  std::shared_ptr<MassRecorder<MR::DisContField>> diag_mass_recorder;
  /// Callback handler to call user defined callbacks.
  SolverCallbackHandler<HWSystem> solver_callback_handler;

protected:
  HWSystem(const LU::SessionReaderSharedPtr &session,
           const SD::MeshGraphSharedPtr &graph);

  /// Bool to enable/disable growth rate recordings
  bool diag_growth_rates_recording_enabled;
  /// Bool to enable/disable mass recordings
  bool diag_mass_recording_enabled;
  /// Hasegawa-Wakatani α
  NekDouble alpha;
  /// Hasegawa-Wakatani κ
  NekDouble kappa;

  virtual void calc_E_and_adv_vels(
      const Array<OneD, const Array<OneD, NekDouble>> &inarray) override final;

  void
  get_phi_solve_rhs(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, NekDouble> &rhs) override final;

  virtual void post_solve() override final;

  virtual void v_GenerateSummary(SU::SummaryList &s) override;
  virtual void v_InitObject(bool DeclareField) override;

  virtual bool v_PostIntegrate(int step) override final;
  virtual bool v_PreIntegrate(int step) override final;
};

} // namespace NESO::Solvers::DriftReduced

#endif // __NESOSOLVERS_DRIFTREDUCED_HWSYSTEM_HPP__