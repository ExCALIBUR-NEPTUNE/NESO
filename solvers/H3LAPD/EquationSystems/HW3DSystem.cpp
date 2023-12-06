#include "HW3DSystem.hpp"
#include "neso_particles.hpp"
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

namespace NESO::Solvers::H3LAPD {
std::string HW3DSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "3DHW", HW3DSystem::create,
        "(3D) Hasegawa-Wakatani equation system as an intermediate step "
        "towards the full H3-LAPD problem");

HW3DSystem::HW3DSystem(const LU::SessionReaderSharedPtr &session,
                       const SD::MeshGraphSharedPtr &graph)
    : UnsteadySystem(session, graph), AdvectionSystem(session, graph),
      DriftReducedSystem(session, graph), m_diff_in_arr(1), m_diff_out_arr(1),
      m_diff_fields(0) {
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
 * electron advection veloctity in v_ExB
 *
 * @param in_arr array of field phys vals
 */
void HW3DSystem::calc_E_and_adv_vels(
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
 *@brief Calculate parallel dynamics term
 *
 * @param in_arr physical values of all fields
 */
void HW3DSystem::calc_par_dyn_term(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr) {

  int npts = GetNpoints();
  int ne_idx = m_field_to_index.get_idx("ne");
  int phi_idx = m_field_to_index.get_idx("phi");

  // Zero temporary out array
  zero_out_array(m_diff_out_arr);

  // Write phi-n into temporary input array
  Vmath::Vsub(npts, m_fields[phi_idx]->GetPhys(), 1, in_arr[ne_idx], 1,
              m_diff_in_arr[0], 1);
  // Use diffusion object to calculate second deriv of phi-n in z direction
  m_diffusion->Diffuse(1, m_diff_fields, m_diff_in_arr, m_diff_out_arr);
  // Multiply by constants to compute term
  Vmath::Smul(npts, m_omega_ce / m_nu_ei, m_diff_out_arr[0], 1, m_par_dyn_term,
              1);
}

/**
 * @brief Populate rhs array ( @p out_arr ) for explicit time integration of
 * the 3D Hasegawa Wakatani equations.
 *
 * @param in_arr physical values of all fields
 * @param[out] out_arr output array (RHSs of time integration equations)
 */
void HW3DSystem::explicit_time_int(
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

  // Add parallel dynamics, implemented as anisotropic parallel-only diffusion
  calc_par_dyn_term(in_arr);
  Vmath::Vsub(npts, out_arr[ne_idx], 1, m_par_dyn_term, 1, out_arr[ne_idx], 1);
  Vmath::Vsub(npts, out_arr[w_idx], 1, m_par_dyn_term, 1, out_arr[w_idx], 1);

  // Add \kappa*\dpartial\phi/\dpartial y to RHS
  Array<OneD, NekDouble> kappa_term(npts);
  m_fields[phi_idx]->PhysDeriv(1, m_fields[phi_idx]->GetPhys(), kappa_term);
  Vmath::Smul(npts, m_kappa, kappa_term, 1, kappa_term, 1);
  Vmath::Vsub(npts, out_arr[ne_idx], 1, kappa_term, 1, out_arr[ne_idx], 1);

  // Add particle sources
  add_particle_sources({"ne"}, out_arr);
}

/**
 * @brief Return the flux vector for the unsteady diffusion problem.
 */
void HW3DSystem::get_flux_vector_diff(
    const Array<OneD, Array<OneD, NekDouble>> &in_arr,
    const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &q_field,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscous_tensor) {
  boost::ignore_unused(in_arr);

  unsigned int nDim = q_field.size();
  unsigned int nConvectiveFields = q_field[0].size();
  unsigned int nPts = q_field[0][0].size();

  // Hard-code coefficients - assumes field in z direction
  NekDouble d[3] = {0.0, 0.0, 1.0};

  for (unsigned int j = 0; j < nDim; ++j) {
    for (unsigned int i = 0; i < nConvectiveFields; ++i) {
      Vmath::Smul(nPts, d[j], q_field[j][i], 1, viscous_tensor[j][i], 1);
    }
  }
}

/**
 * @brief Choose phi solve RHS = w
 *
 * @param in_arr physical values of all fields
 * @param[out] rhs RHS array to pass to Helmsolve
 */
void HW3DSystem::get_phi_solve_rhs(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, NekDouble> &rhs) {
  int npts = GetNpoints();
  int w_idx = m_field_to_index.get_idx("w");
  Vmath::Vcopy(npts, in_arr[w_idx], 1, rhs, 1);
}

/**
 * @brief Read base class params then extra params required for 2D-in-3D HW.
 */
void HW3DSystem::load_params() {
  DriftReducedSystem::load_params();
  // Diffusion type
  m_session->LoadSolverInfo("DiffusionType", m_diff_type, "LDG");

  // kappa (required)
  m_session->LoadParameter("HW_kappa", m_kappa);

  // ω_ce (required)
  m_session->LoadParameter("HW_omega_ce", m_omega_ce);

  // ν_ei (required)
  m_session->LoadParameter("HW_nu_ei", m_nu_ei);
}

/**
 * @brief Post-construction class-initialisation.
 */
void HW3DSystem::v_InitObject(bool DeclareField) {
  DriftReducedSystem::v_InitObject(DeclareField);

  // Bind RHS function for time integration object
  m_ode.DefineOdeRhs(&HW3DSystem::explicit_time_int, this);

  // Set up diffusion object
  m_diffusion =
      SU::GetDiffusionFactory().CreateInstance(m_diff_type, m_diff_type);
  m_diffusion->SetFluxVector(&HW3DSystem::get_flux_vector_diff, this);
  m_diffusion->InitObject(m_session, m_fields);

  // Allocate temporary arrays used in the diffusion calc
  int npts = GetNpoints();
  m_diff_in_arr[0] = Array<OneD, NekDouble>(npts);
  m_diff_out_arr[0] = Array<OneD, NekDouble>(npts, 0.0);
  m_diff_fields[0] = m_fields[m_field_to_index.get_idx("ne")];

  // // Create diagnostic for recording growth rates
  // if (m_diag_growth_rates_recording_enabled) {
  //   m_diag_growth_rates_recorder =
  //       std::make_shared<GrowthRatesRecorder<MultiRegions::DisContField>>(
  //           m_session, m_discont_fields["ne"], m_discont_fields["w"],
  //           m_discont_fields["phi"], GetNpoints(), m_alpha, m_kappa);
  // }

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
bool HW3DSystem::v_PostIntegrate(int step) {
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
bool HW3DSystem::v_PreIntegrate(int step) {
  m_solver_callback_handler.call_pre_integrate(this);

  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recorder->compute_initial_fluid_mass();
  }

  return DriftReducedSystem::v_PreIntegrate(step);
}

} // namespace NESO::Solvers::H3LAPD