#include "LAPDSystem.hpp"
#include <LibUtilities/BasicUtils/Vmath.hpp>

namespace NESO::Solvers::DriftReduced {
std::string LAPDSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "LAPD", LAPDSystem::create, "LAPD equation system");

LAPDSystem::LAPDSystem(const LU::SessionReaderSharedPtr &session,
                       const SD::MeshGraphSharedPtr &graph)
    : DriftReducedSystem(session, graph),
      m_adv_vel_PD(graph->GetSpaceDimension()),
      m_adv_vel_ions(graph->GetSpaceDimension()) {
  this->required_fld_names = {"ne", "Ge", "Gd", "w", "phi"};
  this->int_fld_names = {"ne", "Ge", "Gd", "w"};
}

/**
 * @brief Add collision terms to an output array
 * @param in_arr physical values of all fields
 * @param[out] out_arr array to which collision terms should be added
 */
void LAPDSystem::add_collision_terms(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr) {

  int npts = in_arr[0].size();

  // Field indices
  int Gd_idx = this->field_to_index["Gd"];
  int Ge_idx = this->field_to_index["Ge"];
  int ne_idx = this->field_to_index["ne"];

  /*
  Calculate collision term
  This is the momentum(-density) tranferred from electrons to ions by
  collisions, so add it to Gd rhs, but subtract it from Ge rhs
  */
  Array<OneD, NekDouble> collision_freqs(npts), collision_term(npts),
      vDiffne(npts);
  Vmath::Vmul(npts, in_arr[ne_idx], 1, m_adv_vel_PD[2], 1, vDiffne, 1);
  calc_collision_freqs(in_arr[ne_idx], collision_freqs);
  for (auto ii = 0; ii < npts; ii++) {
    collision_term[ii] = m_me * collision_freqs[ii] * vDiffne[ii];
  }

  // Subtract collision term from Ge rhs
  Vmath::Vsub(npts, out_arr[Ge_idx], 1, collision_term, 1, out_arr[Ge_idx], 1);

  // Add collision term to Gd rhs
  Vmath::Vadd(npts, out_arr[Gd_idx], 1, collision_term, 1, out_arr[Gd_idx], 1);
}

/**
 * @brief Add E_\par terms to an output array
 * @param in_arr physical values of all fields
 * @param[out] out_arr array to which terms should be added
 */
void LAPDSystem::add_E_par_terms(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr) {

  int npts = GetNpoints();

  // Field indices
  int ne_idx = this->field_to_index["ne"];
  int Ge_idx = this->field_to_index["Ge"];
  int Gd_idx = this->field_to_index["Gd"];

  // Calculate EParTerm = e*n_e*EPar (=== e*n_d*EPar)
  // ***Assumes field aligned with z-axis***
  Array<OneD, NekDouble> E_Par_term(npts);
  Vmath::Vmul(npts, in_arr[ne_idx], 1, this->Evec[2], 1, E_Par_term, 1);
  Vmath::Smul(npts, m_charge_e, E_Par_term, 1, E_Par_term, 1);

  // Subtract E_Par_term from out_arr[Ge_idx]
  Vmath::Vsub(npts, out_arr[Ge_idx], 1, E_Par_term, 1, out_arr[Ge_idx], 1);

  // Add E_Par_term to out_arr[Gd_idx]
  Vmath::Vadd(npts, out_arr[Gd_idx], 1, E_Par_term, 1, out_arr[Gd_idx], 1);
}

/**
 * @brief Add \nabla P terms to an output array
 * @param in_arr physical values of all fields
 * @param[out] out_arr array to which terms should be added
 */
void LAPDSystem::add_grad_P_terms(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr) {

  int npts = in_arr[0].size();

  // Field indices
  int ne_idx = this->field_to_index["ne"];
  int Ge_idx = this->field_to_index["Ge"];
  int Gd_idx = this->field_to_index["Gd"];

  // Subtract parallel pressure gradient for Electrons from out_arr[Ge_idx]
  Array<OneD, NekDouble> P_elec(npts), par_gradP_elec(npts);
  Vmath::Smul(npts, m_Te, in_arr[ne_idx], 1, P_elec, 1);
  // ***Assumes field aligned with z-axis***
  m_fields[ne_idx]->PhysDeriv(2, P_elec, par_gradP_elec);
  Vmath::Vsub(npts, out_arr[Ge_idx], 1, par_gradP_elec, 1, out_arr[Ge_idx], 1);

  // Subtract parallel pressure gradient for Ions from out_arr[Ge_idx]
  // N.B. ne === nd
  Array<OneD, NekDouble> P_ions(npts), par_GradP_ions(npts);
  Vmath::Smul(npts, m_Td, in_arr[ne_idx], 1, P_ions, 1);
  // ***Assumes field aligned with z-axis***
  m_fields[ne_idx]->PhysDeriv(2, P_ions, par_GradP_ions);
  Vmath::Vsub(npts, out_arr[Gd_idx], 1, par_GradP_ions, 1, out_arr[Gd_idx], 1);
}

/**
 * @brief Calculate collision frequencies
 *
 * @param ne Array of electron densities
 * @param[out][out] nu_ei Output array for collision frequencies
 */
void LAPDSystem::calc_collision_freqs(const Array<OneD, NekDouble> &ne,
                                      Array<OneD, NekDouble> &nu_ei) {
  Array<OneD, NekDouble> log_lambda(ne.size());
  calc_coulomb_logarithm(ne, log_lambda);
  for (auto ii = 0; ii < ne.size(); ii++) {
    nu_ei[ii] = m_nu_ei_const * ne[ii] * log_lambda[ii];
  }
}

/**
 * @brief Calculate the Coulomb logarithm
 *
 * @param ne Array of electron densities
 * @param[out] log_lambda Output array for Coulomb logarithm values
 */
void LAPDSystem::calc_coulomb_logarithm(const Array<OneD, NekDouble> &ne,
                                        Array<OneD, NekDouble> &log_lambda) {
  /* log_lambda = m_coulomb_log_const - 0.5\ln n_e
       where:
         m_coulomb_log_const = 30 − \ln Z_i +1.5\ln T_e
         n_e in SI units
  */
  for (auto ii = 0; ii < log_lambda.size(); ii++) {
    log_lambda[ii] =
        m_coulomb_log_const - 0.5 * std::log(this->n_to_SI * ne[ii]);
  }
}

/**
 * @brief Compute E = \f$ -\nabla\phi\f$, \f$ v_{E\times B}\f$ and the advection
 * velocities used in the ne/Ge, Gd equations.
 *
 *  @param in_arr physical values of all fields
 */
void LAPDSystem::calc_E_and_adv_vels(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr) {
  DriftReducedSystem::calc_E_and_adv_vels(in_arr);
  int npts = GetNpoints();

  int ne_idx = this->field_to_index["ne"];
  int Gd_idx = this->field_to_index["Gd"];
  int Ge_idx = this->field_to_index["Ge"];

  // v_par,d = Gd / max(ne,n_floor) / md   (N.B. ne === nd)
  for (auto ii = 0; ii < npts; ii++) {
    m_par_vel_ions[ii] =
        in_arr[Gd_idx][ii] /
        std::max(in_arr[ne_idx][ii], this->n_ref * this->n_floor_fac);
  }
  Vmath::Smul(npts, 1.0 / m_md, m_par_vel_ions, 1, m_par_vel_ions, 1);

  // v_par,e = Ge / max(ne,n_floor) / me
  for (auto ii = 0; ii < npts; ii++) {
    this->par_vel_elec[ii] =
        in_arr[Ge_idx][ii] /
        std::max(in_arr[ne_idx][ii], this->n_ref * this->n_floor_fac);
  }
  Vmath::Smul(npts, 1.0 / m_me, this->par_vel_elec, 1, this->par_vel_elec, 1);

  /*
  Store difference in parallel velocities in m_vAdvDiffPar
  N.B. Outer dimension of storage has size ndim to allow it to be used in
  advection operation later
  */
  Vmath::Vsub(npts, this->par_vel_elec, 1, m_par_vel_ions, 1, m_adv_vel_PD[2],
              1);

  // vAdv[iDim] = b[iDim]*v_par + v_ExB[iDim] for each species
  for (auto iDim = 0; iDim < m_graph->GetSpaceDimension(); iDim++) {
    Vmath::Svtvp(npts, b_unit[iDim], this->par_vel_elec, 1, this->ExB_vel[iDim],
                 1, this->adv_vel_elec[iDim], 1);
    Vmath::Svtvp(npts, b_unit[iDim], m_par_vel_ions, 1, this->ExB_vel[iDim], 1,
                 m_adv_vel_ions[iDim], 1);
  }
}

/**
 * @brief Populate rhs array ( @p out_arr ) for explicit time integration of
 * the LAPD equations.
 *
 * @param in_arr physical values of all fields
 * @param[out] out_arr output array (RHSs of time integration equations)
 */
void LAPDSystem::explicit_time_int(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {

  // Zero out_arr
  for (auto ifld = 0; ifld < out_arr.size(); ifld++) {
    Vmath::Zero(out_arr[ifld].size(), out_arr[ifld], 1);
  }

  // Solver for electrostatic potential.
  solve_phi(in_arr);

  // Calculate electric field from Phi, as well as corresponding velocities for
  // all advection operations
  calc_E_and_adv_vels(in_arr);

  // Add advection terms to out_arr, handling (ne, Ge), Gd and w separately
  add_adv_terms({"ne", "Ge"}, this->adv_elec, this->adv_vel_elec, in_arr,
                out_arr, time);
  add_adv_terms({"Gd"}, m_adv_ions, m_adv_vel_ions, in_arr, out_arr, time);
  add_adv_terms({"w"}, this->adv_vort, this->ExB_vel, in_arr, out_arr, time);

  add_grad_P_terms(in_arr, out_arr);

  add_E_par_terms(in_arr, out_arr);

  // Add collision terms to RHS of Ge, Gd eqns
  add_collision_terms(in_arr, out_arr);
  // Add polarisation drift term to vorticity eqn RHS
  add_adv_terms({"ne"}, m_adv_PD, m_adv_vel_PD, in_arr, out_arr, time, {"w"});

  // Add density source via xml-defined function
  add_density_source(out_arr);
}

/**
 * @brief Compute the normal advection velocities for the ion momentum equation
 */
Array<OneD, NekDouble> &LAPDSystem::get_adv_vel_norm_ions() {
  return get_adv_vel_norm(m_norm_vel_ions, m_adv_vel_ions);
}

/**
 * @brief Compute the normal advection velocities for the polarisation drift
 * term.
 */
Array<OneD, NekDouble> &LAPDSystem::get_adv_vel_norm_PD() {
  return get_adv_vel_norm(m_norm_vel_PD, m_adv_vel_PD);
}

/**
 * @brief Compute the flux vector for advection in the ion momentum equation.
 *
 * @param field_vals physical values of all fields
 * @param[out] flux        Resulting flux array
 */
void LAPDSystem::get_flux_vector_ions(
    const Array<OneD, Array<OneD, NekDouble>> &field_vals,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  get_flux_vector(field_vals, m_adv_vel_ions, flux);
}

/**
 * @brief Compute the flux vector for the polarisation drift term.
 *
 * @param field_vals physical values of all fields
 * @param[out] flux        Resulting flux array
 */
void LAPDSystem::get_flux_vector_PD(
    const Array<OneD, Array<OneD, NekDouble>> &field_vals,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  get_flux_vector(field_vals, m_adv_vel_PD, flux);
}

/**
 * @brief Choose phi solve RHS = w * B^2 / (m_d * m_nRef)
 *
 * @param in_arr physical values of all fields
 * @param[out] rhs RHS array to pass to Helmsolve
 */
void LAPDSystem::get_phi_solve_rhs(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, NekDouble> &rhs) {

  int npts = GetNpoints();
  int w_idx = this->field_to_index["w"];
  Vmath::Smul(npts, this->Bmag * this->Bmag / this->n_ref / m_md, in_arr[w_idx],
              1, rhs, 1);
}

/**
 * @brief  Read base class params then extra params required for LAPD.
 */
void LAPDSystem::load_params() {
  DriftReducedSystem::load_params();

  // Factor to convert densities back to SI; used in the Coulomb logarithm calc
  m_session->LoadParameter("ns", this->n_to_SI, 1.0);

  // Charge
  m_session->LoadParameter("e", m_charge_e, 1.0);

  // Ion mass
  m_session->LoadParameter("md", m_md, 2.0);

  // Electron mass - default val is multiplied by 60 to improve convergence
  m_session->LoadParameter("me", m_me, 60. / 1836);

  // Electron temperature in eV
  m_session->LoadParameter("Te", m_Te, 5.0);

  // Ion temperature in eV
  m_session->LoadParameter("Td", m_Td, 0.1);

  // Density independent part of the coulomb logarithm
  m_session->LoadParameter("logLambda_const", m_coulomb_log_const);

  // Pre-factor used when calculating collision frequencies; read from config
  m_session->LoadParameter("nu_ei_const", m_nu_ei_const);
}

/**
 * @brief Post-construction class-initialisation.
 */
void LAPDSystem::v_InitObject(bool DeclareField) {
  DriftReducedSystem::v_InitObject(DeclareField);

  ASSERTL0(m_explicitAdvection,
           "This solver only supports explicit-in-time advection.");

  // Create storage for advection velocities, parallel velocity difference, ExB
  // drift velocity, E field
  int npts = GetNpoints();
  for (int i = 0; i < m_graph->GetSpaceDimension(); ++i) {
    m_adv_vel_ions[i] = Array<OneD, NekDouble>(npts);
    m_adv_vel_PD[i] = Array<OneD, NekDouble>(npts);
    Vmath::Zero(npts, m_adv_vel_PD[i], 1);
  }
  // Create storage for ion parallel velocities
  m_par_vel_ions = Array<OneD, NekDouble>(npts);

  // Define the normal velocity fields.
  // These are populated at each step (by reference) in calls to
  // get_adv_vel_norm()
  if (m_fields[0]->GetTrace()) {
    auto num_trace_pts = GetTraceNpoints();
    m_norm_vel_ions = Array<OneD, NekDouble>(num_trace_pts);
    m_norm_vel_PD = Array<OneD, NekDouble>(num_trace_pts);
  }

  // Advection objects
  m_adv_ions =
      SU::GetAdvectionFactory().CreateInstance(this->adv_type, this->adv_type);
  m_adv_PD =
      SU::GetAdvectionFactory().CreateInstance(this->adv_type, this->adv_type);

  // Set callback functions to compute flux vectors
  m_adv_ions->SetFluxVector(&LAPDSystem::get_flux_vector_ions, this);
  m_adv_PD->SetFluxVector(&LAPDSystem::get_flux_vector_PD, this);

  // Create Riemann solvers (one per advection object) and set normal  velocity
  // callback functions
  m_riemann_ions = SU::GetRiemannSolverFactory().CreateInstance(
      this->riemann_solver_type, m_session);
  m_riemann_ions->SetScalar("Vn", &LAPDSystem::get_adv_vel_norm_ions, this);
  m_riemann_PD = SU::GetRiemannSolverFactory().CreateInstance(
      this->riemann_solver_type, m_session);
  m_riemann_PD->SetScalar("Vn", &LAPDSystem::get_adv_vel_norm_PD, this);

  // Tell advection objects about the Riemann solvers and finish init
  m_adv_ions->SetRiemannSolver(m_riemann_ions);
  m_adv_ions->InitObject(m_session, m_fields);
  m_adv_PD->InitObject(m_session, m_fields);
  m_adv_PD->SetRiemannSolver(m_riemann_PD);

  // Bind RHS function for time integration object
  m_ode.DefineOdeRhs(&LAPDSystem::explicit_time_int, this);
}

} // namespace NESO::Solvers::DriftReduced
