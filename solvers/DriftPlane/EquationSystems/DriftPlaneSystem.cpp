#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

#include "DriftPlaneSystem.hpp"

namespace NESO::Solvers::DriftPlane {
DriftPlaneSystem::DriftPlaneSystem(const LU::SessionReaderSharedPtr &session,
                                   const SD::MeshGraphSharedPtr &graph)
    : TimeEvoEqnSysBase<SU::UnsteadySystem, Particles::EmptyPartSys>(session,
                                                                     graph),
      drift_vel(graph->GetSpaceDimension()) {}

/**
 * @brief Compute the divergence of the sheath closure term.
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

void DriftPlaneSystem::do_ode_projection(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {
  SetBoundaryConditions(time);

  for (int i = 0; i < in_arr.size(); ++i) {
    Vmath::Vcopy(this->n_pts, in_arr[i], 1, out_arr[i], 1);
  }
}

/**
 * @brief Compute the flux vector for this system.
 */
void DriftPlaneSystem::get_flux_vector(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  ASSERTL1(flux[0].size() == this->drift_vel.size(),
           "Dimension of flux array and velocity array do not match");

  int nq = physfield[0].size();

  for (int i = 0; i < flux.size(); ++i) {
    for (int j = 0; j < flux[0].size(); ++j) {
      Vmath::Vmul(nq, physfield[i], 1, this->drift_vel[j], 1, flux[i][j], 1);
    }
  }

  // subtract dn/dy term (this->e is -ve)
  if (this->dndy) {
    for (int i = 0; i < nq; ++i) {
      flux[1][1][i] +=
          this->e * this->T_e / (this->Rxy * this->Rxy) * physfield[0][i];
    }
  }
}

/**
 * @brief Compute the normal advection velocity for this system on the
 * trace/skeleton/edges of the 2D mesh.
 */
Array<OneD, NekDouble> &DriftPlaneSystem::get_normal_velocity() {
  // Number of trace (interface) points
  int num_trace_pts = GetTraceNpoints();

  // Auxiliary variable to compute the normal velocity
  Array<OneD, NekDouble> tmp(num_trace_pts);

  // Reset the normal velocity
  Vmath::Zero(num_trace_pts, this->trace_vnorm, 1);

  // Compute and store dot product of velocity along trace with trace normals
  for (int i = 0; i < this->drift_vel.size(); ++i) {
    m_fields[0]->ExtractTracePhys(this->drift_vel[i], tmp);
    Vmath::Vvtvp(num_trace_pts, m_traceNormals[i], 1, tmp, 1, this->trace_vnorm,
                 1, this->trace_vnorm, 1);
  }

  return this->trace_vnorm;
}

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

  // If ion pressure exists as a field (index !=-1), subtract del^2(ion
  // pressure) from the rhs
  if (ph_idx != -1) {
    // Calc del^2(ion pressure)
    Array<OneD, NekDouble> tempDerivX(this->n_pts, 0.0);
    Array<OneD, NekDouble> tempDerivY(this->n_pts, 0.0);
    Array<OneD, NekDouble> tempLaplacian(this->n_pts, 0.0);
    m_fields[ph_idx]->PhysDeriv(MultiRegions::eX, m_fields[ph_idx]->GetPhys(),
                                tempDerivX);
    m_fields[ph_idx]->PhysDeriv(MultiRegions::eX, tempDerivX, tempDerivX);
    m_fields[ph_idx]->PhysDeriv(MultiRegions::eY, m_fields[ph_idx]->GetPhys(),
                                tempDerivY);
    m_fields[ph_idx]->PhysDeriv(MultiRegions::eY, tempDerivY, tempDerivY);
    Vmath::Vadd(this->n_pts, tempDerivX, 1, tempDerivY, 1, tempLaplacian, 1);

    // Subtract result from rhs
    Vmath::Vsub(this->n_pts, rhs, 1, tempLaplacian, 1, rhs, 1);
  }

  // Set up factors for electrostatic potential solve. We support a
  // generic Helmholtz solve of the form (\nabla^2 - \lambda) u = f, so
  // this sets \lambda to zero.
  StdRegions::ConstFactorMap factors;
  factors[StdRegions::eFactorLambda] = 0.0;
  // Solve for phi. Output of this routine is in coefficient (spectral)
  // space, so backwards transform to physical space since we'll need that
  // for the advection step & computing drift velocity.
  m_fields[phi_idx]->HelmSolve(rhs, m_fields[phi_idx]->UpdateCoeffs(), factors);
  m_fields[phi_idx]->BwdTrans(m_fields[phi_idx]->GetCoeffs(),
                              m_fields[phi_idx]->UpdatePhys());
}

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
 * @param create_field if true, create a new field object and add it to
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
  m_fields[phi_idx] = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
      m_session, m_graph, m_session->GetVariable(phi_idx), true, true);

  // Assign storage for drift velocity.
  for (int i = 0; i < drift_vel.size(); ++i) {
    this->drift_vel[i] = Array<OneD, NekDouble>(this->n_pts);
  }

  // Only DG is supported for now
  NESOASSERT(m_projectionType == MultiRegions::eDiscontinuous,
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
