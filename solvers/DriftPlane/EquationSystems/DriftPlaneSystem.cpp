#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

#include "DriftPlaneSystem.hpp"

namespace MR = Nektar::MultiRegions;

namespace NESO::Solvers::DriftPlane {
DriftPlaneSystem::DriftPlaneSystem(const LU::SessionReaderSharedPtr &session,
                                   const SD::MeshGraphSharedPtr &graph)
    : TimeEvoEqnSysBase<SU::UnsteadySystem, Particles::EmptyPartSys>(session,
                                                                     graph),
      drift_vel(graph->GetSpaceDimension()) {}

/**
 * @brief Compute the divergence of the sheath closure term.
 *
 * @param[in] in_arr physical field values
 * @return Array<OneD, NekDouble> Calculated values of the divergence
 */
Array<OneD, NekDouble> DriftPlaneSystem::calc_div_sheath_closure(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr) {
  int ne_idx = this->field_to_index["ne"];
  int phi_idx = this->field_to_index["phi"];
  Vmath::Vmul(this->n_pts, in_arr[ne_idx], 1, m_fields[phi_idx]->GetPhys(), 1,
              this->div_sheath, 1);
  Vmath::Smul(this->n_pts, 1.0 / this->Lpar, this->div_sheath, 1,
              this->div_sheath, 1);

  return this->div_sheath;
}

/**
 * @brief Compute the drift velocity \f$ \mathbf{v}_{E\times B} \f$.
 */
void DriftPlaneSystem::calc_drift_velocity() {
  int phi_idx = this->field_to_index["phi"];
  m_fields[phi_idx]->PhysDeriv(m_fields[phi_idx]->GetPhys(), this->drift_vel[1],
                               this->drift_vel[0]);
  Vmath::Neg(this->n_pts, this->drift_vel[1], 1);
}

void DriftPlaneSystem::create_riemann_solver() {
  this->riemann_solver = SU::GetRiemannSolverFactory().CreateInstance(
      this->riemann_type, m_session);
}

/**
 * @brief Apply boundary conditions and compute the projection.
 *
 * @param[in] in_arr Given fields.
 * @param[out] out_arr Projected solution.
 * @param[in] time Current time.
 */
void DriftPlaneSystem::do_ode_projection(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {
  SetBoundaryConditions(time);

  // Projection is straight copy in this case.
  for (int ifld = 0; ifld < in_arr.size(); ++ifld) {
    Vmath::Vcopy(this->n_pts, in_arr[ifld], 1, out_arr[ifld], 1);
  }
}

/**
 * @brief Compute the (bulk) flux vector for this system, accounting for dn/dy
 * term.
 *
 * @param[in] phys_vals Physical values of the fields.
 * @param[out] flux Calculated flux vector.
 */
void DriftPlaneSystem::get_flux_vector(
    const Array<OneD, Array<OneD, NekDouble>> &phys_vals,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  ASSERTL1(flux[0].size() == this->drift_vel.size(),
           "Dimension of flux array and velocity array do not match");

  int ne_idx = this->field_to_index["ne"];
  int w_idx = this->field_to_index["w"];

  for (int ifld = 0; ifld < flux.size(); ++ifld) {
    for (int idim = 0; idim < flux[0].size(); ++idim) {
      Vmath::Vmul(phys_vals[ifld].size(), phys_vals[ifld], 1,
                  this->drift_vel[idim], 1, flux[ifld][idim], 1);
    }
  }

  // subtract dn/dy term (this->e is -ve)
  constexpr int y_idx = 1;
  for (int ipt = 0; ipt < phys_vals[ne_idx].size(); ++ipt) {
    flux[w_idx][y_idx][ipt] +=
        this->e * this->T_e / (this->Rxy * this->Rxy) * phys_vals[ne_idx][ipt];
  }
}

/**
 * @brief Compute the normal advection velocity for this system on the
 * trace/skeleton/edges of the 2D mesh.
 *
 * @return Array<OneD, NekDouble>& Normal velocities at the trace points.
 */
Array<OneD, NekDouble> &DriftPlaneSystem::get_normal_velocity() {
  // Number of trace (interface) points
  int num_trace_pts = GetTraceNpoints();

  // Create some space to store the trace phys vals
  Array<OneD, NekDouble> tmp(num_trace_pts);

  // Reset the normal velocities
  Vmath::Zero(num_trace_pts, this->trace_vnorm, 1);

  // Compute and store dot product of velocity along trace with trace normals
  for (int idim = 0; idim < this->drift_vel.size(); ++idim) {
    m_fields[0]->ExtractTracePhys(this->drift_vel[idim], tmp);
    Vmath::Vvtvp(num_trace_pts, m_traceNormals[idim], 1, tmp, 1,
                 this->trace_vnorm, 1, this->trace_vnorm, 1);
  }

  return this->trace_vnorm;
}

/**
 * @brief Function bound as a callback to retrieve trace norm vals in the
 * Riemann solver.
 *
 * @return Array<OneD, NekDouble>& y-components of the trace normals
 */
Array<OneD, NekDouble> &DriftPlaneSystem::get_trace_norm_y() {
  return m_traceNormals[1];
}

/**
 * @brief Load all required session parameters into member variables.
 */
void DriftPlaneSystem::load_params() {
  TimeEvoEqnSysBase<SU::UnsteadySystem, Particles::EmptyPartSys>::load_params();

  m_session->LoadParameter("B", this->B, 0.35);
  m_session->LoadParameter("Lpar", this->Lpar, 1.0);
  m_session->LoadParameter("T_e", this->T_e, 5.0);
  m_session->LoadParameter("Rxy", this->Rxy, 1.5);
  m_session->LoadParameter("e", this->e, -1.0);

  m_session->LoadSolverInfo("AdvectionType", this->adv_type, "WeakDG");
  m_session->LoadSolverInfo("UpwindType", this->riemann_type, "Upwind");
}
void DriftPlaneSystem::solve_phi(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr) {

  int w_idx = this->field_to_index["w"];
  int ph_idx = this->field_to_index.get_idx("ph");
  int phi_idx = this->field_to_index["phi"];

  // Set rhs = B^2 * w
  Array<OneD, NekDouble> rhs(this->n_pts, 0.0);
  Vmath::Smul(this->n_pts, this->B * this->B, in_arr[w_idx], 1, rhs, 1);

  /*
  If ion pressure exists as a field (index !=-1), subtract del^2(ion pressure)
  from the rhs
  */
  if (ph_idx != -1) {
    // Calc del^2(ion pressure)
    Array<OneD, NekDouble> tempDerivX(this->n_pts, 0.0);
    Array<OneD, NekDouble> tempDerivY(this->n_pts, 0.0);
    Array<OneD, NekDouble> tempLaplacian(this->n_pts, 0.0);
    m_fields[ph_idx]->PhysDeriv(MR::eX, m_fields[ph_idx]->GetPhys(),
                                tempDerivX);
    m_fields[ph_idx]->PhysDeriv(MR::eX, tempDerivX, tempDerivX);
    m_fields[ph_idx]->PhysDeriv(MR::eY, m_fields[ph_idx]->GetPhys(),
                                tempDerivY);
    m_fields[ph_idx]->PhysDeriv(MR::eY, tempDerivY, tempDerivY);
    Vmath::Vadd(this->n_pts, tempDerivX, 1, tempDerivY, 1, tempLaplacian, 1);

    // Subtract result from rhs
    Vmath::Vsub(this->n_pts, rhs, 1, tempLaplacian, 1, rhs, 1);
  }

  /*
   Set up factors for electrostatic potential solve.
   Helmholtz solve has the form (\nabla^2 - \lambda) u = f, so set \lambda=0
  */
  StdRegions::ConstFactorMap factors;
  factors[StdRegions::eFactorLambda] = 0.0;
  /*
   Solve for phi. Output of the routine is in coefficient (spectral) space.
   Backwards transform to physical space since phi is needed to compute v_ExB.
  */
  m_fields[phi_idx]->HelmSolve(rhs, m_fields[phi_idx]->UpdateCoeffs(), factors);
  m_fields[phi_idx]->BwdTrans(m_fields[phi_idx]->GetCoeffs(),
                              m_fields[phi_idx]->UpdatePhys());
}

/**
 * @brief Add Driftplane params to the eqn sys summary.
 *
 * @param[inout] s The summary list to modify
 */
void DriftPlaneSystem::v_GenerateSummary(SU::SummaryList &s) {
  UnsteadySystem::v_GenerateSummary(s);

  SU::AddSummaryItem(s, "|B|", this->B);
  SU::AddSummaryItem(s, "e", this->e);
  SU::AddSummaryItem(s, "Lpar", this->Lpar);
  SU::AddSummaryItem(s, "Rxy", this->Rxy);
  SU::AddSummaryItem(s, "T_e", this->T_e);

  SU::AddSummaryItem(s, "AdvectionType", this->adv_type);
  SU::AddSummaryItem(s, "UpwindType", this->riemann_type);
}

/**
 * @brief Post-construction class initialisation.
 *
 * @param[in] create_field if true, create a new field object and add it to
 * m_fields. Optional, defaults to true.
 */
void DriftPlaneSystem::v_InitObject(bool create_field) {

  TimeEvoEqnSysBase::v_InitObject(create_field);

  // Check variables are in the expected order
  check_var_idx(0, "ne");
  check_var_idx(1, "w");
  check_var_idx(2, "phi");

  int ne_idx = this->field_to_index["ne"];
  int w_idx = this->field_to_index["w"];
  int phi_idx = this->field_to_index["phi"];

  // Set up storage for sheath divergence.
  this->div_sheath = Array<OneD, NekDouble>(this->n_pts, 0.0);

  // Make phi a ContField
  m_fields[phi_idx] = Nektar::MemoryManager<MR::ContField>::AllocateSharedPtr(
      m_session, m_graph, m_session->GetVariable(phi_idx), true, true);

  // Assign storage for drift velocity.
  for (int idim = 0; idim < drift_vel.size(); ++idim) {
    this->drift_vel[idim] = Array<OneD, NekDouble>(this->n_pts);
  }

  // Only DG is supported for now
  NESOASSERT(m_projectionType == MR::eDiscontinuous,
             "Unsupported projection type: only DG currently supported.");

  // Do not forwards transform ICs
  m_homoInitialFwd = false;

  // Define the normal velocity fields
  if (m_fields[0]->GetTrace()) {
    this->trace_vnorm = Array<OneD, NekDouble>(GetTraceNpoints());
  }

  // Create an advection object of the required type
  this->adv_obj =
      SU::GetAdvectionFactory().CreateInstance(this->adv_type, this->adv_type);

  // Set callback func used by the advection object to obtain the flux vector
  this->adv_obj->SetFluxVector(&DriftPlaneSystem::get_flux_vector, this);

  // Create the Riemann solver
  create_riemann_solver();
  this->riemann_solver->SetScalar("Vn", &DriftPlaneSystem::get_normal_velocity,
                                  this);

  // Tell the advection object about the Riemann solver to use,
  // and then get it set up.
  this->adv_obj->SetRiemannSolver(this->riemann_solver);
  this->adv_obj->InitObject(m_session, m_fields);

  NESOASSERT(m_explicitAdvection,
             "This solver only supports explicit-in-time advection.");

  m_ode.DefineProjection(&DriftPlaneSystem::do_ode_projection, this);
}

} // namespace NESO::Solvers::DriftPlane
