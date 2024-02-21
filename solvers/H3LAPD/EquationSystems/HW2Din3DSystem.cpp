#include "HW2Din3DSystem.hpp"
#include "neso_particles.hpp"
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

namespace NESO::Solvers::H3LAPD {
std::string HW2Din3DSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "2Din3DHW", HW2Din3DSystem::create,
        "(2D) Hasegawa-Wakatani equation system as an intermediate step "
        "towards the full H3-LAPD problem");

HW2Din3DSystem::HW2Din3DSystem(const LU::SessionReaderSharedPtr &session,
                               const SD::MeshGraphSharedPtr &graph)
    : UnsteadySystem(session, graph), AdvectionSystem(session, graph),
      DriftReducedSystem(session, graph), HWSystem(session, graph) {}

/**
 * @brief Populate rhs array ( @p out_arr ) for explicit time integration of
 * the 2D Hasegawa Wakatani equations.
 *
 * @param in_arr physical values of all fields
 * @param[out] out_arr output array (RHSs of time integration equations)
 */
void HW2Din3DSystem::explicit_time_int(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {

  // Check in_arr for NaNs
  for (auto &var : {"ne", "w"}) {
    auto fidx = m_field_to_index.get_idx(var);
    for (auto ii = 0; ii < in_arr[fidx].size(); ii++) {
      std::stringstream err_msg;
      err_msg << "Found NaN in field " << var;
      NESOASSERT(std::isfinite(in_arr[fidx][ii]), err_msg.str().c_str());
    }
  }

  zero_out_array(out_arr);

  // Solve for electrostatic potential
  solve_phi(in_arr);

  // Calculate electric field from Phi, as well as corresponding drift velocity
  calc_E_and_adv_vels(in_arr);

  // Get field indices
  int npts = GetNpoints();
  int ne_idx = m_field_to_index.get_idx("ne");
  int phi_idx = m_field_to_index.get_idx("phi");
  int w_idx = m_field_to_index.get_idx("w");

  // Advect ne and w (m_adv_vel_elec === m_ExB_vel for HW)
  add_adv_terms({"ne"}, m_adv_elec, m_adv_vel_elec, in_arr, out_arr, time);
  add_adv_terms({"w"}, m_adv_vort, m_ExB_vel, in_arr, out_arr, time);

  // Add \alpha*(\phi-n_e) to RHS
  Array<OneD, NekDouble> HWterm_2D_alpha(npts);
  Vmath::Vsub(npts, m_fields[phi_idx]->GetPhys(), 1,
              m_fields[ne_idx]->GetPhys(), 1, HWterm_2D_alpha, 1);
  Vmath::Smul(npts, m_alpha, HWterm_2D_alpha, 1, HWterm_2D_alpha, 1);
  Vmath::Vadd(npts, out_arr[w_idx], 1, HWterm_2D_alpha, 1, out_arr[w_idx], 1);
  Vmath::Vadd(npts, out_arr[ne_idx], 1, HWterm_2D_alpha, 1, out_arr[ne_idx], 1);

  // Add \kappa*\dpartial\phi/\dpartial y to RHS
  Array<OneD, NekDouble> HWterm_2D_kappa(npts);
  m_fields[phi_idx]->PhysDeriv(1, m_fields[phi_idx]->GetPhys(),
                               HWterm_2D_kappa);
  Vmath::Smul(npts, m_kappa, HWterm_2D_kappa, 1, HWterm_2D_kappa, 1);
  Vmath::Vsub(npts, out_arr[ne_idx], 1, HWterm_2D_kappa, 1, out_arr[ne_idx], 1);

  // Add particle sources
  add_particle_sources({"ne"}, out_arr);
}

/**
 * @brief Read base class params then extra params required for 2D-in-3D HW.
 */
void HW2Din3DSystem::load_params() {
  DriftReducedSystem::load_params();

  // alpha (required)
  m_session->LoadParameter("HW_alpha", m_alpha);

  // kappa (required)
  m_session->LoadParameter("HW_kappa", m_kappa);
}

/**
 * @brief Post-construction class-initialisation.
 */
void HW2Din3DSystem::v_InitObject(bool DeclareField) {
  HWSystem::v_InitObject(DeclareField);

  // Bind RHS function for time integration object
  m_ode.DefineOdeRhs(&HW2Din3DSystem::explicit_time_int, this);

  // Create diagnostic for recording growth rates
  if (m_diag_growth_rates_recording_enabled) {
    m_diag_growth_rates_recorder =
        std::make_shared<GrowthRatesRecorder<MultiRegions::DisContField>>(
            m_session, 2, m_discont_fields["ne"], m_discont_fields["w"],
            m_discont_fields["phi"], GetNpoints(), m_alpha, m_kappa);
  }
}

} // namespace NESO::Solvers::H3LAPD
