#include "HW2DSystem.hpp"
#include "neso_particles.hpp"
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

namespace NESO::Solvers::DriftReduced {
std::string HW2DSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "2DHW", HW2DSystem::create,
        "(2D) Hasegawa-Wakatani equation system as an intermediate step "
        "towards the full H3-LAPD problem");

HW2DSystem::HW2DSystem(const LU::SessionReaderSharedPtr &session,
                       const SD::MeshGraphSharedPtr &graph)
    : HWSystem(session, graph) {}

/**
 * @brief Populate rhs array ( @p out_arr ) for explicit time integration of
 * the 2D Hasegawa Wakatani equations.
 *
 * @param in_arr physical values of all fields
 * @param[out] out_arr output array (RHSs of time integration equations)
 */
void HW2DSystem::explicit_time_int(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {

#ifdef NESO_DEBUG
  /**
   * Check in_arr for NaN/inf (*very* slow).
   * N.B. A similar check in UnsteadySystem::v_DoSolve only catches NaN.
   */
  for (auto &var : {"ne", "w"}) {
    auto fidx = this->field_to_index[var];
    for (auto ii = 0; ii < in_arr[fidx].size(); ii++) {
      std::stringstream err_msg;
      err_msg << "Found NaN/inf in field " << var;
      NESOASSERT(std::isfinite(in_arr[fidx][ii]), err_msg.str().c_str());
    }
  }
#endif

  zero_out_array(out_arr);

  // Solve for electrostatic potential
  solve_phi(in_arr);

  // Calculate electric field from Phi, as well as corresponding drift velocity
  calc_E_and_adv_vels(in_arr);

  // Get field indices
  int npts = GetNpoints();
  int ne_idx = this->field_to_index["ne"];
  int phi_idx = this->field_to_index["phi"];
  int w_idx = this->field_to_index["w"];

  // Advect ne and w (adv_vel_elec === ExB_vel for HW)
  add_adv_terms({"ne"}, this->adv_elec, this->adv_vel_elec, in_arr, out_arr,
                time);
  add_adv_terms({"w"}, this->adv_vort, this->ExB_vel, in_arr, out_arr, time);

  // Add \alpha*(\phi-n_e) to RHS
  Array<OneD, NekDouble> HWterm_2D_alpha(npts);
  Vmath::Vsub(npts, m_fields[phi_idx]->GetPhys(), 1,
              m_fields[ne_idx]->GetPhys(), 1, HWterm_2D_alpha, 1);
  Vmath::Smul(npts, this->alpha, HWterm_2D_alpha, 1, HWterm_2D_alpha, 1);
  Vmath::Vadd(npts, out_arr[w_idx], 1, HWterm_2D_alpha, 1, out_arr[w_idx], 1);
  Vmath::Vadd(npts, out_arr[ne_idx], 1, HWterm_2D_alpha, 1, out_arr[ne_idx], 1);

  // Add \kappa*\dpartial\phi/\dpartial y to RHS
  Array<OneD, NekDouble> HWterm_2D_kappa(npts);
  m_fields[phi_idx]->PhysDeriv(1, m_fields[phi_idx]->GetPhys(),
                               HWterm_2D_kappa);
  Vmath::Smul(npts, this->kappa, HWterm_2D_kappa, 1, HWterm_2D_kappa, 1);
  Vmath::Vsub(npts, out_arr[ne_idx], 1, HWterm_2D_kappa, 1, out_arr[ne_idx], 1);

  // Add particle sources
  if (this->particles_enabled) {
    add_particle_sources({"ne"}, out_arr);
  }
}

/**
 * @brief Read base class params then extra params required for 2D-in-3D HW.
 */
void HW2DSystem::load_params() {
  DriftReducedSystem::load_params();

  // alpha (required)
  m_session->LoadParameter("HW_alpha", this->alpha);

  // kappa (required)
  m_session->LoadParameter("HW_kappa", this->kappa);
}

/**
 * @brief Post-construction class-initialisation.
 */
void HW2DSystem::v_InitObject(bool DeclareField) {
  HWSystem::v_InitObject(DeclareField);

  // Bind RHS function for time integration object
  m_ode.DefineOdeRhs(&HW2DSystem::explicit_time_int, this);

  // Create diagnostic for recording growth rates
  if (this->diag_growth_rates_recording_enabled) {
    this->diag_growth_rates_recorder =
        std::make_shared<GrowthRatesRecorder<MR::DisContField>>(
            m_session, 2, this->discont_fields["ne"], this->discont_fields["w"],
            this->discont_fields["phi"], GetNpoints(), this->alpha,
            this->kappa);
  }
}

} // namespace NESO::Solvers::DriftReduced
