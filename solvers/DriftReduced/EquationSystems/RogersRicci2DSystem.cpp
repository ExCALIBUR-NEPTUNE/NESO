#include "RogersRicci2DSystem.hpp"

#include <MultiRegions/ContField.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

using Nektar::Array;
using Nektar::NekDouble;
using Nektar::OneD;

namespace NESO::Solvers::DriftReduced {

std::string RogersRicci2D::className =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "RogersRicci2D", RogersRicci2D::create,
        "System for the Rogers-Ricci 2D system of equations.");

RogersRicci2D::RogersRicci2D(
    const Nektar::LibUtilities::SessionReaderSharedPtr &session,
    const Nektar::SpatialDomains::MeshGraphSharedPtr &graph)
    : DriftReducedSystem(session, graph) {}

/**
 * @brief Choose phi solve RHS = w
 *
 * @param in_arr physical values of all fields
 * @param[out] rhs RHS array to pass to Helmsolve
 */
void RogersRicci2D::get_phi_solve_rhs(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, NekDouble> &rhs) {
  int w_idx = this->field_to_index["w"];
  Vmath::Smul(this->n_pts, 1.0, in_arr[w_idx], 1, rhs, 1);
}

void RogersRicci2D::v_InitObject(bool DeclareField) {
  DriftReducedSystem::v_InitObject(DeclareField);

  NESOASSERT(m_fields.size() == 4,
             "Incorrect number of variables detected (expected 4): check your "
             "session file.");

  // Store mesh dimension for easy retrieval later.
  NESOASSERT(this->n_dims == 2 || this->n_dims == 3,
             "Solver only supports 2D or 3D meshes.");

  // Check variable order is as expected
  check_var_idx(n_idx, "n");
  check_var_idx(Te_idx, "T_e");
  check_var_idx(w_idx, "w");
  check_var_idx(phi_idx, "phi");

  m_fields[phi_idx] = Nektar::MemoryManager<MR::ContField>::AllocateSharedPtr(
      m_session, m_graph, m_session->GetVariable(phi_idx), true, true);
  m_intVariables = {n_idx, Te_idx, w_idx};

  switch (m_projectionType) {
  case MR::eDiscontinuous: {
    m_homoInitialFwd = false;

    if (m_fields[0]->GetTrace()) {
      this->trace_norm_vels = Array<OneD, NekDouble>(GetTraceNpoints());
    }

    std::string adv_name, riemann_type;
    m_session->LoadSolverInfo("AdvectionType", adv_name, "WeakDG");
    this->adv_obj =
        SU::GetAdvectionFactory().CreateInstance(adv_name, adv_name);
    this->adv_obj->SetFluxVector(&RogersRicci2D::get_flux_vector, this);

    m_session->LoadSolverInfo("RiemannType", riemann_type, "Upwind");
    this->riemann_solver =
        SU::GetRiemannSolverFactory().CreateInstance(riemann_type, m_session);
    this->riemann_solver->SetScalar("Vn", &RogersRicci2D::get_norm_vel, this);

    // Tell advection object which Riemann solver to use and do initialisation
    this->adv_obj->SetRiemannSolver(this->riemann_solver);
    this->adv_obj->InitObject(m_session, m_fields);
    break;
  }

  default: {
    NESOASSERT(false, "Unsupported projection type: only discontinuous"
                      " projection supported.");
    break;
  }
  }

  m_ode.DefineOdeRhs(&RogersRicci2D::explicit_time_int, this);
  // N.B. projection already bound to DriftReduced::do_ode_projection

  if (!m_explicitAdvection) {
    this->implicit_helper = std::make_shared<ImplicitHelper>(
        m_session, m_fields, m_ode, m_intVariables.size());
    this->implicit_helper->init_non_lin_sys_solver();
    m_ode.DefineImplicitSolve(&ImplicitHelper::implicit_time_int,
                              this->implicit_helper);
  }

  // Store distance of quad points from origin in transverse plane.
  // (used to compute source terms)
  Array<OneD, NekDouble> x = Array<OneD, NekDouble>(this->n_pts);
  Array<OneD, NekDouble> y = Array<OneD, NekDouble>(this->n_pts);
  this->r = Array<OneD, NekDouble>(this->n_pts);
  if (this->n_dims == 3) {
    Array<OneD, NekDouble> z = Array<OneD, NekDouble>(this->n_pts);
    m_fields[0]->GetCoords(x, y, z);
  } else {
    m_fields[0]->GetCoords(x, y);
  }
  for (auto ipt = 0; ipt < this->n_pts; ++ipt) {
    this->r[ipt] = sqrt(x[ipt] * x[ipt] + y[ipt] * y[ipt]);
  }
}

/**
 * @brief Evaluate the right-hand side of the ODE system used to integrate in
 * time.
 *
 * This routine performs the bulk of the work in this class, and essentially
 * computes the right hand side term of the generalised ODE system
 *
 * \f\[ \frac{\partial \mathbf{u}}{\partial t} = \mathbf{R}(\mathbf{u}) \f\]
 *
 * The order of operations is as follows:
 *
 * - First, compute the electrostatic potential \f$ \phi \f$
 * - Using this, compute the drift velocity \f$ (\partial_y\phi,
 *   -\partial_x\phi).
 * - Then evaluate the \f$ \nabla\cdot\mathbf{F} \f$ operator using the
 *   advection object
 * - Finally put this on the RHS and evaluate the source terms for each field.
 *
 * @param inarray    Array containing each field's current state.
 * @param outarray   The result of the right-hand side operator for each field
 *                   being time integrated.
 * @param time       Current value of time.
 */
void RogersRicci2D::explicit_time_int(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {

  solve_phi(in_arr);

  // Calculate electric field from Phi, then v_ExB
  calc_E_and_adv_vels(in_arr);

  // Advect all fields up to, but not including, the electric potential
  this->adv_obj->Advect(phi_idx, m_fields, this->ExB_vel, in_arr, out_arr,
                        time);

  // Convenient references to field vals
  Array<OneD, NekDouble> n = in_arr[n_idx];
  Array<OneD, NekDouble> T_e = in_arr[Te_idx];
  Array<OneD, NekDouble> w = in_arr[w_idx];
  Array<OneD, NekDouble> phi = m_fields[phi_idx]->UpdatePhys();

  // Add remaining terms to RHS arrays
  for (auto ipt = 0; ipt < this->n_pts; ++ipt) {
    // Exponential term that features in all three time evo equations
    NekDouble exp_term = exp(
        this->coulomb_log - phi[ipt] / sqrt(T_e[ipt] * T_e[ipt] + this->T_eps));

    // Source term (same for n and T in scaled units)
    NekDouble src_term =
        0.03 *
        (1.0 - tanh((this->rho_s0 * this->r[ipt] - this->r_s) / this->L_s));

    // Compile RHS arrays
    out_arr[n_idx][ipt] =
        -40 * out_arr[n_idx][ipt] - 1.0 / 24.0 * exp_term * n[ipt] + src_term;
    out_arr[Te_idx][ipt] = -40 * out_arr[Te_idx][ipt] -
                           1.0 / 36.0 * (1.71 * exp_term - 0.71) * T_e[ipt] +
                           src_term;
    out_arr[w_idx][ipt] =
        -40 * out_arr[w_idx][ipt] + 1.0 / 24.0 * (1 - exp_term);
  }
}

/**
 * @brief Compute the flux vector for this system.
 */
void RogersRicci2D::get_flux_vector(
    const Array<OneD, Array<OneD, NekDouble>> &field_vals,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  DriftReducedSystem::get_flux_vector(field_vals, this->ExB_vel, flux);
}

/**
 * @brief Compute the normal advection velocity for this system on the
 * trace/skeleton/edges of the 2D mesh.
 */
Array<OneD, NekDouble> &RogersRicci2D::get_norm_vel() {
  return DriftReducedSystem::get_adv_vel_norm(this->trace_norm_vels,
                                              this->ExB_vel);
}

/**
 * @brief Read model params required for the 2D Rogers & Ricci system.
 */
void RogersRicci2D::load_params() {
  DriftReducedSystem::load_params();

  // Temp local vars
  NekDouble Ls_boost, rs_norm;

  // Coulomb log (optional)
  m_session->LoadParameter("coulomb_log", this->coulomb_log, 3.0);

  // Boost factor for L_s (optional)
  m_session->LoadParameter("Ls_boost", Ls_boost, 2.0);

  // rho_s0 (Also space normalisation;optional)
  m_session->LoadParameter("rho_s0", this->rho_s0, 1.2e-2);

  // Source scale length in normalised units (optional)
  m_session->LoadParameter("rs_norm", rs_norm, 20.0);

  // Regularisation value used in 1/T term (optional)
  m_session->LoadParameter("T_eps", this->T_eps, 1e-4);

  // Set source scale lengths
  this->L_s = 0.5 * this->rho_s0 * Ls_boost;
  this->r_s = rs_norm * this->rho_s0;
}

} // namespace NESO::Solvers::DriftReduced
