#include "RogersRicci2D.h"

#include <MultiRegions/ContField.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

using namespace Nektar;

namespace NESO::Solvers::H3LAPD {

std::string RogersRicci2D::className =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "RogersRicci2D", RogersRicci2D::create,
        "System for the Rogers-Ricci 2D system of equations.");

RogersRicci2D::RogersRicci2D(const LU::SessionReaderSharedPtr &session,
                             const SpatialDomains::MeshGraphSharedPtr &graph)
    : DriftReducedSystem(session, graph) {}

void check_var_idx(const LU::SessionReaderSharedPtr session, const int &idx,
                   const std::string var_name) {
  std::stringstream err;
  err << "Expected variable index " << idx << " to correspond to '" << var_name
      << "'. Check your session file.";
  NESOASSERT(session->GetVariable(idx).compare(var_name) == 0, err.str());
}

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
  check_var_idx(m_session, n_idx, "n");
  check_var_idx(m_session, Te_idx, "T_e");
  check_var_idx(m_session, w_idx, "w");
  check_var_idx(m_session, phi_idx, "phi");

  m_fields[phi_idx] = MemoryManager<MR::ContField>::AllocateSharedPtr(
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

    // Tell the advection object about the Riemann solver to use, and
    // then get it set up.
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
  m_ode.DefineProjection(&RogersRicci2D::do_ode_projection, this);

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

  // Factors for Helmsolve
  StdRegions::ConstFactorMap factors;
  factors[StdRegions::eFactorLambda] = 0.0;
  if (this->n_dims == 3) {
    factors[StdRegions::eFactorCoeffD22] = 0.0;
  }

  Vmath::Zero(m_fields[phi_idx]->GetNcoeffs(),
              m_fields[phi_idx]->UpdateCoeffs(), 1);

  // Poisson solve for electric potential
  m_fields[phi_idx]->HelmSolve(in_arr[w_idx], m_fields[phi_idx]->UpdateCoeffs(),
                               factors);
  // Output is in coefficient space; back transform to get physical values
  m_fields[phi_idx]->BwdTrans(m_fields[phi_idx]->GetCoeffs(),
                              m_fields[phi_idx]->UpdatePhys());

  // Calculate electric field from Phi, then v_ExB
  calc_E_and_adv_vels(in_arr);

  // Advect all fields up to, but not including, the electric potential
  this->adv_obj->Advect(phi_idx, m_fields, this->ExB_vel, in_arr, out_arr,
                        time);

  // Params
  const NekDouble rho_s0 = 1.2e-2;
  const NekDouble r_s = 20 * rho_s0;
  const NekDouble Ls_boost = 2.0;
  const NekDouble L_s = 0.5 * rho_s0 * Ls_boost;
  const NekDouble T_eps = 1e-4;
  const NekDouble coulomb_log = 3.0;

  // Convenient references to field vals
  Array<OneD, NekDouble> n = in_arr[n_idx];
  Array<OneD, NekDouble> T_e = in_arr[Te_idx];
  Array<OneD, NekDouble> w = in_arr[w_idx];
  Array<OneD, NekDouble> phi = m_fields[phi_idx]->UpdatePhys();

  // Add remaining terms to RHS arrays
  for (auto ipt = 0; ipt < this->n_pts; ++ipt) {
    // Exponential term that features in all three time evo equations
    NekDouble exp_term =
        exp(coulomb_log - phi[ipt] / sqrt(T_e[ipt] * T_e[ipt] + T_eps));

    // Source term (same for n and T in scaled units)
    NekDouble src_term =
        0.03 * (1.0 - tanh((rho_s0 * this->r[ipt] - r_s) / L_s));

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
 * @brief Perform projection into correct polynomial space.
 *
 * This routine projects the @p in_arr input and ensures the @p out_arr output
 * lives in the correct space. Since we are hard-coding DG, this corresponds to
 * a simple copy from in to out, since no elemental connectivity is required and
 * the output of the RHS function is polynomial.
 */
void RogersRicci2D::do_ode_projection(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {
  SetBoundaryConditions(time);

  for (auto ifld = 0; ifld < in_arr.size(); ++ifld) {
    Vmath::Vcopy(this->n_pts, in_arr[ifld], 1, out_arr[ifld], 1);
  }
}

/**
 * @brief Compute the flux vector for this system.
 */
void RogersRicci2D::get_flux_vector(
    const Array<OneD, Array<OneD, NekDouble>> &field_vals,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  NESOASSERT(flux[0].size() <= this->ExB_vel.size(),
             "Dimension of flux array must be less than or equal to that of "
             "the drift velocity array.");

  for (auto ifld = 0; ifld < flux.size(); ++ifld) {
    for (int idim = 0; idim < flux[0].size(); ++idim) {
      Vmath::Vmul(this->n_pts, field_vals[ifld], 1, this->ExB_vel[idim], 1,
                  flux[ifld][idim], 1);
    }
  }
}

/**
 * @brief Compute the normal advection velocity for this system on the
 * trace/skeleton/edges of the 2D mesh.
 */
Array<OneD, NekDouble> &RogersRicci2D::get_norm_vel() {
  // Number of trace (interface) points
  int nTracePts = GetTraceNpoints();

  // Auxiliary variable to compute the normal velocity
  Array<OneD, NekDouble> tmp(nTracePts);

  // Reset the normal velocity
  Vmath::Zero(nTracePts, this->trace_norm_vels, 1);

  // Compute dot product of velocity along trace with trace normals. Store in
  // this->trace_norm_vels.
  for (auto idim = 0; idim < this->n_dims; ++idim) {
    m_fields[0]->ExtractTracePhys(this->ExB_vel[idim], tmp);

    Vmath::Vvtvp(nTracePts, m_traceNormals[idim], 1, tmp, 1,
                 this->trace_norm_vels, 1, this->trace_norm_vels, 1);
  }

  return this->trace_norm_vels;
}

} // namespace NESO::Solvers::H3LAPD
