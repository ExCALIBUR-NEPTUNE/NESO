#ifndef H3LAPD_HW3D_SYSTEM_H
#define H3LAPD_HW3D_SYSTEM_H

#include "nektar_interface/utilities.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/Diffusion/Diffusion.h>
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
 * @brief 3D Hasegawa-Wakatani equation system.
 * @details Intended as an intermediate step towards the full LAPD equation
 * system. Evolves ne, w, phi only, no momenta, no ions.
 */
class HW3DSystem : virtual public DriftReducedSystem {
public:
  friend class MemoryManager<HW3DSystem>;

  /**
   * @brief Creates an instance of this class.
   */
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<HW3DSystem>::AllocateSharedPtr(session, graph);
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
  SolverCallbackHandler<HW3DSystem> m_solver_callback_handler;

protected:
  HW3DSystem(const LU::SessionReaderSharedPtr &session,
             const SD::MeshGraphSharedPtr &graph);

  void
  calc_par_dyn_term(const Array<OneD, const Array<OneD, NekDouble>> &in_arr);

  virtual void calc_E_and_adv_vels(
      const Array<OneD, const Array<OneD, NekDouble>> &inarray) override;

  void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray,
                    const NekDouble time) override;

  void get_flux_vector_diff(
      const Array<OneD, Array<OneD, NekDouble>> &in_arr,
      const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &q_field,
      Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscous_tensor);

  void
  get_phi_solve_rhs(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, NekDouble> &rhs) override;

  void load_params() override;

  virtual void v_InitObject(bool DeclareField) override;

  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;

private:
  /// Bool to enable/disable growth rate recordings
  bool m_diag_growth_rates_recording_enabled;
  /// Bool to enable/disable mass recordings
  bool m_diag_mass_recording_enabled;

  // Diffussion type
  std::string m_diff_type;

  // Diffusion object
  SU::DiffusionSharedPtr m_diffusion;

  /**
   * Hasegawa-Wakatani κ - not clear whether this should be included in 3D or
   * not
   */
  NekDouble m_kappa;
  /// Electron-ion collision frequency
  NekDouble m_nu_ei;
  /// Cyclotron frequency for electrons
  NekDouble m_omega_ce;

  // Array for storage of parallel dynamics term
  Array<OneD, NekDouble> m_par_dyn_term;

  // Arrays to store temporary fields and values for the diffusion operation
  Array<OneD, MultiRegions::ExpListSharedPtr> m_diff_fields;
  Array<OneD, Array<OneD, NekDouble>> m_diff_in_arr, m_diff_out_arr;
};

} // namespace NESO::Solvers::H3LAPD
#endif // H3LAPD_HW3D_SYSTEM_H