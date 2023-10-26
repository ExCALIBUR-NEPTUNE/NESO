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

/**
 * @brief 2D Hasegawa-Wakatani equation system designed to work in a 3D domain.
 * @details Intended as an intermediate step towards the full LAPD equation
 * system. Evolves ne, w, phi only, no momenta, no ions.
 */
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