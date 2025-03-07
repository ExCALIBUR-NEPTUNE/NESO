#include "Blob2DSystem.hpp"
#include "../RiemannSolvers/DriftUpwindSolver.hpp"

namespace NESO::Solvers::DriftPlane {

std::string Blob2DSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "Blob2D", Blob2DSystem::create,
        "System for the Blob2D drift-plane equations.");

Blob2DSystem::Blob2DSystem(const LU::SessionReaderSharedPtr &session,
                           const SD::MeshGraphSharedPtr &graph)
    : DriftPlaneSystem(session, graph) {
  // Fields are (electron) number density, vorticity, potential
  this->required_fld_names = {"ne", "w", "phi"};
  // Time-integrate density and vorticity
  this->int_fld_names = {"ne", "w"};
  this->dndy = true;
}

/**
 * @brief Construct the RiemannSolver object and setup callbacks.
 *
 */
void Blob2DSystem::create_riemann_solver() {
  if (this->dndy) {
    this->riemann_solver = std::make_shared<DriftUpwindSolver>(
        m_session, -this->e * this->T_e / (this->Rxy * this->Rxy),
        this->field_to_index["ne"], this->field_to_index["w"]);
    this->riemann_solver->SetScalar("ny", &Blob2DSystem::get_trace_norm_y,
                                    this);
  } else {
    DriftPlaneSystem::create_riemann_solver();
  }
}

/**
 * @brief Compute the RHS terms.
 *
 * @param[in] in_arr Input field values
 * @param[out] out_arr RHS values
 * @param[in] time Current time
 */
void Blob2DSystem::explicit_time_int(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {
  const int ne_idx = this->field_to_index["ne"];
  const int w_idx = this->field_to_index["w"];
  const int phi_idx = this->field_to_index["phi"];

  // Solve for electrostatic potential and obtain corresponding drift velocity
  solve_phi(in_arr);
  calc_drift_velocity();

  // Calculate divergence of the sheath closure
  calc_div_sheath_closure(in_arr);

  // Advect quantities at the drift velocity
  this->adv_obj->Advect(m_intVariables.size(), m_fields, this->drift_vel,
                        in_arr, out_arr, time);

  // Put advection term on the right hand side
  for (int i = 0; i < out_arr.size(); ++i) {
    Vmath::Smul(this->n_pts, -1.0 / this->B, out_arr[i], 1, out_arr[i], 1);
  }

  // Add ne source term
  Vmath::Svtvp(this->n_pts, 1.0 / this->e, this->div_sheath, 1, out_arr[ne_idx],
               1, out_arr[ne_idx], 1);

  // We assume b is a constant and so curl is zero; so just add the sheath term
  Vmath::Vadd(this->n_pts, out_arr[w_idx], 1, this->div_sheath, 1,
              out_arr[w_idx], 1);

  if (!this->dndy) {
    // compute dn/dy
    Array<OneD, NekDouble> tmp(this->n_pts, 0.0);
    m_fields[ne_idx]->PhysDeriv(MultiRegions::eY, in_arr[ne_idx], tmp);

    /*
     Diamagnetic drift term
     DM: Looks like this should be this->e instead of -this->e, but then this
      advects in the wrong direction; possibly extra minus sign from curl(b/B)
     term?
     */
    Vmath::Svtvp(this->n_pts, -this->e * this->T_e / (this->Rxy * this->Rxy),
                 tmp, 1, out_arr[w_idx], 1, out_arr[w_idx], 1);
  }
}

/**
 * @brief Post-construction class initialisation.
 *
 * @param[in] create_field if true, create a new field object and add it to
 * m_fields. Optional, defaults to true.
 */
void Blob2DSystem::v_InitObject(bool create_fields) {
  DriftPlaneSystem::v_InitObject(create_fields);

  m_ode.DefineOdeRhs(&Blob2DSystem::explicit_time_int, this);
}

} // namespace NESO::Solvers::DriftPlane
