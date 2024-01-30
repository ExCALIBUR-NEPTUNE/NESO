#include "HWSystem.hpp"
#include "neso_particles.hpp"
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

namespace NESO::Solvers::H3LAPD {

HWSystem::HWSystem(const LU::SessionReaderSharedPtr &session,
                   const SD::MeshGraphSharedPtr &graph)
    : UnsteadySystem(session, graph), AdvectionSystem(session, graph),
      DriftReducedSystem(session, graph) {
  m_required_flds = {"ne", "w", "phi"};
  m_int_fld_names = {"ne", "w"};

  // Frequency of growth rate recording. Set zero to disable.
  m_diag_growth_rates_recording_enabled =
      session->DefinesParameter("growth_rates_recording_step");

  // Frequency of mass recording. Set zero to disable.
  m_diag_mass_recording_enabled =
      session->DefinesParameter("mass_recording_step");
}

/**
 * @brief Override DriftReducedSystem::calc_E_and_adv_vels in order to set
 * electron advection velocity in v_ExB
 *
 * @param in_arr array of field phys vals
 */
void HWSystem::calc_E_and_adv_vels(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr) {
  DriftReducedSystem::calc_E_and_adv_vels(in_arr);
  int npts = GetNpoints();

  Vmath::Zero(npts, m_par_vel_elec, 1);
  // vAdv[iDim] = b[iDim]*v_par + v_ExB[iDim] for each species
  for (auto iDim = 0; iDim < m_graph->GetSpaceDimension(); iDim++) {
    Vmath::Svtvp(npts, m_b_unit[iDim], m_par_vel_elec, 1, m_ExB_vel[iDim], 1,
                 m_adv_vel_elec[iDim], 1);
  }
}

/**
 * @brief Choose phi solve RHS = w
 *
 * @param in_arr physical values of all fields
 * @param[out] rhs RHS array to pass to Helmsolve
 */
void HWSystem::get_phi_solve_rhs(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, NekDouble> &rhs) {
  int npts = GetNpoints();
  int w_idx = m_field_to_index.get_idx("w");
  Vmath::Vcopy(npts, in_arr[w_idx], 1, rhs, 1);
}

void HWSystem::v_GenerateSummary(SU::SummaryList &s) {
  DriftReducedSystem::v_GenerateSummary(s);
  SU::AddSummaryItem(s, "HW alpha", m_alpha);
  SU::AddSummaryItem(s, "HW kappa", m_kappa);
}

/**
 * @brief Post-construction class-initialisation.
 */
void HWSystem::v_InitObject(bool DeclareField) {
  DriftReducedSystem::v_InitObject(DeclareField);

  // Create diagnostic for recording fluid and particles masses
  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recorder =
        std::make_shared<MassRecorder<MultiRegions::DisContField>>(
            m_session, m_particle_sys, m_discont_fields["ne"]);
  }
}

/**
 * @brief Compute diagnostics, if enabled, then call base class member func.
 */
bool HWSystem::v_PostIntegrate(int step) {
  if (m_diag_growth_rates_recording_enabled) {
    m_diag_growth_rates_recorder->compute(step);
  }

  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recorder->compute(step);
  }

  m_solver_callback_handler.call_post_integrate(this);
  return DriftReducedSystem::v_PostIntegrate(step);
}

/**
 * @brief Do initial set up for mass recording diagnostic (first call only), if
 * enabled, then call base class member func.
 */
bool HWSystem::v_PreIntegrate(int step) {
  m_solver_callback_handler.call_pre_integrate(this);

  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recorder->compute_initial_fluid_mass();
  }

  return DriftReducedSystem::v_PreIntegrate(step);
}

} // namespace NESO::Solvers::H3LAPD
