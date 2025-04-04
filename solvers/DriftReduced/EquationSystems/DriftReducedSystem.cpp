#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

#include "DriftReducedSystem.hpp"

namespace NESO::Solvers::DriftReduced {
DriftReducedSystem::DriftReducedSystem(
    const LU::SessionReaderSharedPtr &session,
    const SD::MeshGraphSharedPtr &graph)
    : TimeEvoEqnSysBase<SU::UnsteadySystem, NeutralParticleSystem>(session,
                                                                   graph) {}

/**
 * @brief Compute advection terms and add them to an output array
 * @details For each field listed in @p field_names copy field pointers from
 * m_fields and physical values from @p in_arr to temporary arrays and create a
 * temporary output array of the same size. Call Advect() on @p adv_obj passing
 * the temporary arrays. Finally, loop over @p eqn_labels to determine which
 * element(s) of @p out_arr to subtract the results from.
 *
 * N.B. The creation of temporary arrays is necessary to bypass restrictions in
 * the Nektar advection API.
 *
 * @param field_names List of field names to compute advection terms for
 * @param adv_obj Nektar advection object
 * @param adv_vel Array of advection velocities (outer dim size = nfields)
 * @param in_arr Physical values for *all* fields
 * @param[out] out_arr RHS array (for *all* fields)
 * @param time Simulation time
 * @param eqn_labels List of field names identifying indices in out_arr to add
 * the result to. Defaults to @p field_names
 */
void DriftReducedSystem::add_adv_terms(
    std::vector<std::string> field_names, const SU::AdvectionSharedPtr adv_obj,
    const Array<OneD, Array<OneD, NekDouble>> &adv_vel,
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time,
    std::vector<std::string> eqn_labels) {

  // Default is to add result of advecting field f to the RHS of df/dt equation
  if (eqn_labels.empty()) {
    eqn_labels = std::vector(field_names);
  } else {
    ASSERTL1(field_names.size() == eqn_labels.size(),
             "add_adv_terms: Number of quantities being advected must match "
             "the number of equation labels.");
  }

  int nfields = field_names.size();
  int npts = GetNpoints();

  /* Make temporary copies of target fields, in_arr vals and initialise a
   * temporary output array
   */
  Array<OneD, MR::ExpListSharedPtr> tmp_fields(nfields);
  Array<OneD, Array<OneD, NekDouble>> tmp_inarray(nfields);
  Array<OneD, Array<OneD, NekDouble>> tmp_outarray(nfields);
  for (auto ii = 0; ii < nfields; ii++) {
    int idx = this->field_to_index[field_names[ii]];
    tmp_fields[ii] = m_fields[idx];
    tmp_inarray[ii] = Array<OneD, NekDouble>(npts);
    Vmath::Vcopy(npts, in_arr[idx], 1, tmp_inarray[ii], 1);
    tmp_outarray[ii] = Array<OneD, NekDouble>(out_arr[idx].size());
  }
  // Compute advection terms; result is returned in temporary output array
  adv_obj->Advect(tmp_fields.size(), tmp_fields, adv_vel, tmp_inarray,
                  tmp_outarray, time);

  // Subtract temporary output array from the appropriate indices of out_arr
  for (auto ii = 0; ii < nfields; ii++) {
    int idx = this->field_to_index[eqn_labels[ii]];
    Vmath::Vsub(out_arr[idx].size(), out_arr[idx], 1, tmp_outarray[ii], 1,
                out_arr[idx], 1);
  }
}

/**
 * @brief Add (density) source term via a Nektar session function.
 *
 * @details Looks for a function called "dens_src", evaluates it, and adds the
 * result to @p out_arr
 *
 * @param[out] out_arr RHS array to add the source too
 * @todo Check function exists, rather than relying on Nektar ASSERT
 */
void DriftReducedSystem::add_density_source(
    Array<OneD, Array<OneD, NekDouble>> &out_arr) {

  int ne_idx = this->field_to_index["ne"];
  int npts = GetNpoints();
  Array<OneD, NekDouble> tmpx(npts), tmpy(npts), tmpz(npts);
  m_fields[ne_idx]->GetCoords(tmpx, tmpy, tmpz);
  Array<OneD, NekDouble> dens_src(npts, 0.0);
  LU::EquationSharedPtr dens_src_func =
      m_session->GetFunction("dens_src", ne_idx);
  dens_src_func->Evaluate(tmpx, tmpy, tmpz, dens_src);
  Vmath::Vadd(npts, out_arr[ne_idx], 1, dens_src, 1, out_arr[ne_idx], 1);
}

/**
 * @brief Adds particle sources.
 * @details For each <field_name> in @p target_fields , look for another field
 * called <field_name>_src. If it exists, add the physical values of
 * field_name_src to the appropriate index of @p out_arr.
 *
 * @param target_fields list of Nektar field names for which to look for a
 * '_src' counterpart
 *  @param[out] out_arr      the RHS array
 *
 */
void DriftReducedSystem::add_particle_sources(
    std::vector<std::string> target_fields,
    Array<OneD, Array<OneD, NekDouble>> &out_arr) {
  for (auto target_field : target_fields) {
    int src_field_idx = this->field_to_index[target_field + "_src"];

    if (src_field_idx >= 0) {
      // Check that the target field is one that is time integrated
      auto tmp_it = std::find(this->int_fld_names.cbegin(),
                              this->int_fld_names.cend(), target_field);
      ASSERTL0(tmp_it != this->int_fld_names.cend(),
               "Target for particle source ['" + target_field +
                   "'] does not appear in the list of time-integrated fields "
                   "(int_fld_names).")
      /*
      N.B. out_arr can be smaller than m_fields if any fields aren't
      time-integrated, so can't just use out_arr_idx =
      this->field_to_index[target_field]
       */
      auto out_arr_idx = std::distance(this->int_fld_names.cbegin(), tmp_it);
      Vmath::Vadd(out_arr[out_arr_idx].size(), out_arr[out_arr_idx], 1,
                  m_fields[src_field_idx]->GetPhys(), 1, out_arr[out_arr_idx],
                  1);
    }
  }
}

/**
 *  @brief Compute E = \f$ -\nabla\phi\f$, \f$ v_{E\times B}\f$ and the ExB
 * drift velocity.
 *
 * @param in_arr array of field phys vals
 */
void DriftReducedSystem::calc_E_and_adv_vels(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr) {
  int phi_idx = this->field_to_index["phi"];
  int npts = GetNpoints();

  if (this->n_dims == 3) {
    m_fields[phi_idx]->PhysDeriv(m_fields[phi_idx]->GetPhys(), this->Evec[0],
                                 this->Evec[1], Evec[2]);
    Vmath::Neg(npts, this->Evec[0], 1);
    Vmath::Neg(npts, this->Evec[1], 1);
    Vmath::Neg(npts, this->Evec[2], 1);
  } else {
    m_fields[phi_idx]->PhysDeriv(m_fields[phi_idx]->GetPhys(), this->Evec[0],
                                 this->Evec[1]);
    Vmath::Neg(npts, this->Evec[0], 1);
    Vmath::Neg(npts, this->Evec[1], 1);
  }

  // v_ExB = this->Evec x Bvec / |B|^2
  Vmath::Svtsvtp(npts, this->Bvec[2] / this->Bmag / this->Bmag, this->Evec[1],
                 1, -this->Bvec[1] / this->Bmag / this->Bmag, this->Evec[2], 1,
                 this->ExB_vel[0], 1);
  Vmath::Svtsvtp(npts, this->Bvec[0] / this->Bmag / this->Bmag, this->Evec[2],
                 1, -this->Bvec[2] / this->Bmag / this->Bmag, this->Evec[0], 1,
                 this->ExB_vel[1], 1);
  if (this->n_dims == 3) {
    Vmath::Svtsvtp(npts, this->Bvec[1] / this->Bmag / this->Bmag, this->Evec[0],
                   1, -this->Bvec[0] / this->Bmag / this->Bmag, this->Evec[1],
                   1, this->ExB_vel[2], 1);
  }
}

/**
 * @brief Perform projection into correct polynomial space.
 *
 * @details This routine projects the @p in_arr input and ensures the @p
 * out_arr output lives in the correct space. Since we are hard-coding DG, this
 * corresponds to a simple copy from in to out, since no elemental
 * connectivity is required and the output of the RHS function is
 * polynomial.
 *
 * @param in_arr Unprojected values
 * @param[out] out_arr Projected values
 * @param time Current simulation time
 *
 */
void DriftReducedSystem::do_ode_projection(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {
  int num_vars = in_arr.size();

  /**
   * this->n_pts is used here, rather than in_arr[i].size(), to workaround the
   * current behaviour of ImplicitHelper. It uses Nektar::Array's
   * overloaded + operator to get offset regions of unrolled arrays of size
   * n_pts*num_vars. This results in the sizes of in_arr elements being
   * [n_pts*num_vars,n_pts*num_vars-1 ... n_pts] rather than
   * [n_pts,n_pts...n_pts], with elements from n_pts+1 onwards being irrelevant.
   */
  for (int i = 0; i < num_vars; ++i) {
    Vmath::Vcopy(this->n_pts, in_arr[i], 1, out_arr[i], 1);
  }
}

/**
 *  @brief Compute components of advection velocities normal to trace elements
 * (faces, in 3D).
 *
 * @param[in,out] trace_vel_norm Trace normal velocities for each field
 * @param         adv_vel        Advection velocities for each field
 */
Array<OneD, NekDouble> &DriftReducedSystem::get_adv_vel_norm(
    Array<OneD, NekDouble> &trace_vel_norm,
    const Array<OneD, Array<OneD, NekDouble>> &adv_vel) {

  NESOASSERT(adv_vel.size() >= m_traceNormals.size(),
             "DriftReducedSystem::get_adv_vel_norm: adv_vel array must have "
             "dimension at least as large as m_traceNormals.");
  // Number of trace (interface) points
  int num_trace_pts = GetTraceNpoints();
  // Auxiliary variable to compute normal velocities
  Array<OneD, NekDouble> tmp(num_trace_pts);

  // Zero previous values
  Vmath::Zero(num_trace_pts, trace_vel_norm, 1);

  //  Compute dot product of advection velocity with the trace normals and store
  for (int i = 0; i < m_traceNormals.size(); ++i) {
    m_fields[0]->ExtractTracePhys(adv_vel[i], tmp);
    Vmath::Vvtvp(num_trace_pts, m_traceNormals[i], 1, tmp, 1, trace_vel_norm, 1,
                 trace_vel_norm, 1);
  }
  return trace_vel_norm;
}

/**
 *  @brief Compute trace-normal advection velocities for the electron density.
 */
Array<OneD, NekDouble> &DriftReducedSystem::get_adv_vel_norm_elec() {
  return get_adv_vel_norm(this->norm_vel_elec, this->adv_vel_elec);
}

/**
 * @brief Compute trace-normal advection velocities for the vorticity equation.
 */
Array<OneD, NekDouble> &DriftReducedSystem::get_adv_vel_norm_vort() {
  return get_adv_vel_norm(this->norm_vel_vort, this->ExB_vel);
}

/**
 *  @brief Construct flux array.
 *
 * @param  field_vals Physical values for each advection field
 * @param  adv_vel    Advection velocities for each advection field
 * @param[out] flux       Flux array
 */
void DriftReducedSystem::get_flux_vector(
    const Array<OneD, Array<OneD, NekDouble>> &field_vals,
    const Array<OneD, Array<OneD, NekDouble>> &adv_vel,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  ASSERTL1(flux[0].size() == adv_vel.size(),
           "Dimension of flux array and advection velocity array do not match");
  int npts = field_vals[0].size();

  for (auto i = 0; i < flux.size(); ++i) {
    for (auto j = 0; j < flux[0].size(); ++j) {
      Vmath::Vmul(npts, field_vals[i], 1, adv_vel[j], 1, flux[i][j], 1);
    }
  }
}

/**
 * @brief Construct the flux vector for the diffusion problem.
 * @todo not implemented
 */
void DriftReducedSystem::get_flux_vector_diff(
    const Array<OneD, Array<OneD, NekDouble>> &in_arr,
    const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &q_field,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscous_tensor) {
  std::cout << "*** GetFluxVectorDiff not defined! ***" << std::endl;
}

/**
 * @brief Compute the flux vector for advection in the electron density and
 * momentum equations.
 *
 * @param field_vals   Array of Fields ptrs
 * @param[out] flux         Resulting flux array
 */
void DriftReducedSystem::get_flux_vector_elec(
    const Array<OneD, Array<OneD, NekDouble>> &field_vals,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  get_flux_vector(field_vals, this->adv_vel_elec, flux);
}

/**
 * @brief Compute the flux vector for advection in the vorticity equation.
 *
 * @param field_vals   Array of Fields ptrs
 * @param[out] flux        Resulting flux array
 */
void DriftReducedSystem::get_flux_vector_vort(
    const Array<OneD, Array<OneD, NekDouble>> &field_vals,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  // Advection velocity is v_ExB in the vorticity equation
  get_flux_vector(field_vals, this->ExB_vel, flux);
}

/**
 * @brief Load all required session parameters into member variables.
 */
void DriftReducedSystem::load_params() {
  TimeEvoEqnSysBase<SU::UnsteadySystem, NeutralParticleSystem>::load_params();

  // Type of advection to use -- in theory we also support flux reconstruction
  // for quad-based meshes, or you can use a standard convective term if you
  // were fully continuous in space. Default is DG.
  m_session->LoadSolverInfo("AdvectionType", this->adv_type, "WeakDG");

  // ***Assumes field aligned with z-axis***
  // Magnetic field strength. Fix B = [0, 0, Bxy] for now
  this->Bvec = std::vector<NekDouble>(m_graph->GetSpaceDimension(), 0);
  m_session->LoadParameter("Bxy", this->Bvec[2], 0.1);

  // Coefficient factors for potential solve
  m_session->LoadParameter("d00", this->d00, 1);
  m_session->LoadParameter("d11", this->d11, 1);
  m_session->LoadParameter("d22", this->d22, 1);

  // Factor to set density floor; default to 1e-5 (Hermes-3 default)
  m_session->LoadParameter("n_floor_fac", this->n_floor_fac, 1e-5);

  // Reference number density
  m_session->LoadParameter("nRef", this->n_ref, 1.0);

  // Type of Riemann solver to use. Default = "Upwind"
  m_session->LoadSolverInfo("UpwindType", this->riemann_solver_type, "Upwind");

  // Particle-related parameters
  m_session->LoadParameter("num_particle_steps_per_fluid_step",
                           this->num_part_substeps, 1);
  m_session->LoadParameter("particle_num_write_particle_steps",
                           this->num_write_particle_steps, 0);
  this->part_timestep = m_timestep / this->num_part_substeps;

  // Compute some properties derived from params
  this->Bmag =
      std::sqrt(this->Bvec[0] * this->Bvec[0] + this->Bvec[1] * this->Bvec[1] +
                this->Bvec[2] * this->Bvec[2]);
  this->b_unit = std::vector<NekDouble>(m_graph->GetSpaceDimension());
  for (auto idim = 0; idim < this->b_unit.size(); idim++) {
    this->b_unit[idim] = (this->Bmag > 0) ? this->Bvec[idim] / this->Bmag : 0.0;
  }
}

/**
 * @brief Calls HelmSolve to solve for the electric potential, given the
 * right-hand-side returned by get_phi_solve_rhs
 *
 * @param in_arr Array of physical field values
 */
void DriftReducedSystem::solve_phi(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr) {

  // Field indices
  int npts = GetNpoints();
  int phi_idx = this->field_to_index["phi"];

  // Define rhs
  Array<OneD, NekDouble> rhs(npts);
  get_phi_solve_rhs(in_arr, rhs);

  // Set up factors for electrostatic potential solve
  StdRegions::ConstFactorMap factors;
  // Helmholtz => Poisson (lambda = 0)
  factors[StdRegions::eFactorLambda] = 0.0;
  // Set coefficient factors
  factors[StdRegions::eFactorCoeffD00] = this->d00;
  factors[StdRegions::eFactorCoeffD11] = this->d11;
  if (this->n_dims == 3) {
    factors[StdRegions::eFactorCoeffD22] = this->d22;
  }

  // Solve for phi. Output of this routine is in coefficient (spectral)
  // space, so backwards transform to physical space since we'll need that
  // for the advection step & computing drift velocity.
  m_fields[phi_idx]->HelmSolve(rhs, m_fields[phi_idx]->UpdateCoeffs(), factors);
  m_fields[phi_idx]->BwdTrans(m_fields[phi_idx]->GetCoeffs(),
                              m_fields[phi_idx]->UpdatePhys());
}

void DriftReducedSystem::v_GenerateSummary(SU::SummaryList &s) {
  UnsteadySystem::v_GenerateSummary(s);

  std::stringstream tmpss;
  tmpss << "[" << this->d00 << "," << this->d11;
  if (this->n_dims == 3) {
    tmpss << "," << this->d22;
  }
  tmpss << "]";
  SU::AddSummaryItem(s, "Helmsolve coeffs.", tmpss.str());

  SU::AddSummaryItem(s, "Reference density", this->n_ref);
  SU::AddSummaryItem(s, "Density floor", this->n_floor_fac);

  SU::AddSummaryItem(s, "Riemann solver", this->riemann_solver_type);
  // Particle stuff
  SU::AddSummaryItem(s, "Num. part. substeps", this->num_part_substeps);
  SU::AddSummaryItem(s, "Part. output freq", this->num_write_particle_steps);
  tmpss = std::stringstream();
  tmpss << "[" << this->Bvec[0] << "," << this->Bvec[1] << "," << this->Bvec[2]
        << "]";
  SU::AddSummaryItem(s, "B", tmpss.str());
  SU::AddSummaryItem(s, "|B|", this->Bmag);
}

/**
 * @brief Post-construction class initialisation.
 *
 * @param create_field if true, create a new field object and add it to
 * m_fields. Optional, defaults to true.
 */
void DriftReducedSystem::v_InitObject(bool create_field) {
  if (this->particles_enabled) {
    this->required_fld_names.push_back("ne_src");
  }
  TimeEvoEqnSysBase::v_InitObject(create_field);

  NESOASSERT(this->n_dims == 2 || this->n_dims == 3,
             "2DHW system requires a 2D or 3D mesh.");

  // Since we are starting from a setup where each field is defined to be a
  // discontinuous field (and thus support DG), the first thing we do is to
  // recreate the phi field so that it is continuous, in order to support the
  // Poisson solve. Note that you can still perform a Poisson solve using a
  // discontinuous field, which is done via the hybridisable discontinuous
  // Galerkin (HDG) approach.
  int phi_idx = this->field_to_index["phi"];
  m_fields[phi_idx] = Nektar::MemoryManager<MR::ContField>::AllocateSharedPtr(
      m_session, m_graph, m_session->GetVariable(phi_idx), true, true);

  // Create storage for advection velocities, parallel velocity difference,ExB
  // drift velocity, E field. These are 3D regardless of the mesh dimension.
  int npts = GetNpoints();
  for (auto idim = 0; idim < 3; ++idim) {
    this->adv_vel_elec[idim] = Array<OneD, NekDouble>(npts, 0.0);
    this->ExB_vel[idim] = Array<OneD, NekDouble>(npts, 0.0);
    this->Evec[idim] = Array<OneD, NekDouble>(npts, 0.0);
  }

  // Evec has 3 dimensions regardless of mesh dimension (simplifies ExB calc)
  for (int i = 0; i < Evec.size(); ++i) {
    this->Evec[i] = Array<OneD, NekDouble>(npts, 0.0);
  }

  // Create storage for electron parallel velocities
  this->par_vel_elec = Array<OneD, NekDouble>(npts);

  // Type of advection class to be used. By default, we only support the
  // discontinuous projection, since this is the only approach we're
  // considering for this solver.
  ASSERTL0(m_projectionType == MR::eDiscontinuous,
           "Unsupported projection type: only discontinuous"
           " projection supported.");

  // Do not forwards transform initial condition.
  m_homoInitialFwd = false; ////

  // Define the normal velocity fields.
  // These are populated at each step (by reference) in calls to GetVnAdv()
  if (m_fields[0]->GetTrace()) {
    auto nTrace = GetTraceNpoints();
    this->norm_vel_elec = Array<OneD, NekDouble>(nTrace);
    this->norm_vel_vort = Array<OneD, NekDouble>(nTrace);
  }

  // Advection objects
  // Need one per advection velocity
  this->adv_elec =
      SU::GetAdvectionFactory().CreateInstance(this->adv_type, this->adv_type);
  this->adv_vort =
      SU::GetAdvectionFactory().CreateInstance(this->adv_type, this->adv_type);

  // Set callback functions to compute flux vectors
  this->adv_elec->SetFluxVector(&DriftReducedSystem::get_flux_vector_elec,
                                this);
  this->adv_vort->SetFluxVector(&DriftReducedSystem::get_flux_vector_vort,
                                this);

  // Create Riemann solvers (one per advection object) and set normal velocity
  // callback functions
  this->riemann_elec = SU::GetRiemannSolverFactory().CreateInstance(
      this->riemann_solver_type, m_session);
  this->riemann_elec->SetScalar(
      "Vn", &DriftReducedSystem::get_adv_vel_norm_elec, this);
  this->riemann_vort = SU::GetRiemannSolverFactory().CreateInstance(
      this->riemann_solver_type, m_session);
  this->riemann_vort->SetScalar(
      "Vn", &DriftReducedSystem::get_adv_vel_norm_vort, this);

  // Tell advection objects about the Riemann solvers and finish init
  this->adv_elec->SetRiemannSolver(this->riemann_elec);
  this->adv_elec->InitObject(m_session, m_fields);
  this->adv_vort->SetRiemannSolver(this->riemann_vort);
  this->adv_vort->InitObject(m_session, m_fields);

  // Bind projection function for time integration object
  m_ode.DefineProjection(&DriftReducedSystem::do_ode_projection, this);

  // Store DisContFieldSharedPtr casts of fields in a map, indexed by name, for
  // use in particle project,evaluate operations
  int idx = 0;
  for (auto &field_name : m_session->GetVariables()) {
    this->discont_fields[field_name] =
        std::dynamic_pointer_cast<MR::DisContField>(m_fields[idx]);
    idx++;
  }

  if (this->particles_enabled) {
    // Set up object to project onto density source field
    int low_order_project;
    m_session->LoadParameter("low_order_project", low_order_project, 0);
    if (low_order_project) {
      ASSERTL0(
          this->discont_fields.count("ne_src_interp"),
          "Intermediate, lower order interpolation field not found in config.");
      this->particle_sys->setup_project(this->discont_fields["ne_src_interp"],
                                        this->discont_fields["ne_src"]);
    } else {
      this->particle_sys->setup_project(this->discont_fields["ne_src"]);
    }

    // Set up object to evaluate density field
    this->particle_sys->setup_evaluate_ne(this->discont_fields["ne"]);
  }
}

/**
 * @brief Override v_PostIntegrate to do particle output
 *
 * @param step Time step number
 */
bool DriftReducedSystem::v_PostIntegrate(int step) {
  // Writes a step of the particle trajectory.
  if (this->particles_enabled && this->particle_sys->is_output_step(step)) {
    this->particle_sys->write(step);
    this->particle_sys->write_source_fields();
  }
  return UnsteadySystem::v_PostIntegrate(step);
}

/**
 * @brief Override v_PreIntegrate to do particle system integration, projection
 * onto source terms.
 *
 * @param step Time step number
 */
bool DriftReducedSystem::v_PreIntegrate(int step) {
  if (this->particles_enabled) {
    // Integrate the particle system to the requested time.
    this->particle_sys->integrate(m_time + m_timestep, this->part_timestep);
    // Project onto the source fields
    this->particle_sys->project_source_terms();
  }

  return UnsteadySystem::v_PreIntegrate(step);
}

/**
 * @brief Convenience function to zero a Nektar Array of 1D Arrays.
 *
 * @param out_arr Array of 1D arrays to be zeroed
 *
 */
void DriftReducedSystem::zero_out_array(
    Array<OneD, Array<OneD, NekDouble>> &out_arr) {
  for (auto ifld = 0; ifld < out_arr.size(); ifld++) {
    Vmath::Zero(out_arr[ifld].size(), out_arr[ifld], 1);
  }
}
} // namespace NESO::Solvers::DriftReduced
