#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

#include "SOLSystem.hpp"

namespace NESO::Solvers::SimpleSOL {

std::string SOLSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "SimpleSOL", SOLSystem::create,
        "SOL equations in conservative variables.");

SOLSystem::SOLSystem(const LU::SessionReaderSharedPtr &session,
                     const SD::MeshGraphSharedPtr &graph)
    : TimeEvoEqnSysBase<SU::UnsteadySystem, NeutralParticleSystem>(session,
                                                                   graph) {

  // m_spacedim isn't set at this point, for some reason; use mesh dim instead
  NESOASSERT(graph->GetSpaceDimension() == 1 || graph->GetSpaceDimension() == 2,
             "Unsupported mush dimension for SOLSystem - must be 1 or 2.");
  if (graph->GetSpaceDimension() == 2) {
    this->required_fld_names = {"rho", "rhou", "rhov", "E"};
  } else {
    this->required_fld_names = {"rho", "rhou", "E"};
  }
  int_fld_names = std::vector<std::string>(this->required_fld_names);
}

/**
 * @brief Initialization object for SOLSystem class.
 */
void SOLSystem::v_InitObject(bool DeclareField) {
  TimeEvoEqnSysBase<SU::UnsteadySystem, NeutralParticleSystem>::v_InitObject(
      DeclareField);

  for (int i = 0; i < m_fields.size(); i++) {
    // Use BwdTrans to make sure initial condition is in solution space
    m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(), m_fields[i]->UpdatePhys());
  }

  this->var_converter =
      Nektar::MemoryManager<VariableConverter>::AllocateSharedPtr(m_session,
                                                                  m_spacedim);

  ASSERTL0(m_session->DefinesSolverInfo("UPWINDTYPE"),
           "No UPWINDTYPE defined in session.");

  // Store velocity field indices in the format required by the Riemann solver.
  this->vel_fld_indices = Array<OneD, Array<OneD, NekDouble>>(1);
  this->vel_fld_indices[0] = Array<OneD, NekDouble>(m_spacedim);
  for (int i = 0; i < m_spacedim; ++i) {
    this->vel_fld_indices[0][i] = 1 + i;
  }

  // Loading parameters from session file
  m_session->LoadParameter("Gamma", this->gamma, 1.4);

  // Setting up advection and diffusion operators
  init_advection();

  // Set up forcing/source term objects.
  this->fluid_src_terms = SU::Forcing::Load(m_session, shared_from_this(),
                                            m_fields, m_fields.size());

  m_ode.DefineOdeRhs(&SOLSystem::explicit_time_int, this);
  m_ode.DefineProjection(&SOLSystem::do_ode_projection, this);
}

/**
 * @brief Initialisation, including creation of advection object.
 */
void SOLSystem::init_advection() {
  // Only DG is supported
  ASSERTL0(m_projectionType == MR::eDiscontinuous,
           "Unsupported projection type: must be DG.");

  std::string adv_type, riemann_type;
  m_session->LoadSolverInfo("AdvectionType", adv_type, "WeakDG");

  this->adv_obj = SU::GetAdvectionFactory().CreateInstance(adv_type, adv_type);

  this->adv_obj->SetFluxVector(&SOLSystem::get_flux_vector, this);

  // Setting up Riemann solver for advection operator
  m_session->LoadSolverInfo("UpwindType", riemann_type, "Average");

  SU::RiemannSolverSharedPtr riemann_solver;
  riemann_solver =
      SU::GetRiemannSolverFactory().CreateInstance(riemann_type, m_session);

  // Setting up parameters for advection operator Riemann solver
  riemann_solver->SetParam("gamma", &SOLSystem::get_gamma, this);
  riemann_solver->SetAuxVec("vecLocs", &SOLSystem::get_vec_locs, this);
  riemann_solver->SetVector("N", &SOLSystem::get_trace_norms, this);

  // Concluding initialisation of advection / diffusion operators
  this->adv_obj->SetRiemannSolver(riemann_solver);
  this->adv_obj->InitObject(m_session, m_fields);
}

/**
 * @brief Compute the right-hand side.
 */
void SOLSystem::explicit_time_int(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {
  int num_vars = in_arr.size();
  int num_pts = GetNpoints();
  int num_trace_pts = GetTraceTotPoints();

  // Store forwards/backwards values along trace space
  Array<OneD, Array<OneD, NekDouble>> fwd(num_vars);
  Array<OneD, Array<OneD, NekDouble>> bwd(num_vars);

  if (m_HomogeneousType == eHomogeneous1D) {
    fwd = NullNekDoubleArrayOfArray;
    bwd = NullNekDoubleArrayOfArray;
  } else {
    for (int i = 0; i < num_vars; ++i) {
      fwd[i] = Array<OneD, NekDouble>(num_trace_pts, 0.0);
      bwd[i] = Array<OneD, NekDouble>(num_trace_pts, 0.0);
      m_fields[i]->GetFwdBwdTracePhys(in_arr[i], fwd[i], bwd[i]);
    }
  }

  // Calculate advection
  do_advection(in_arr, out_arr, time, fwd, bwd);

  // Ensure advection terms have the correct sign
  for (int i = 0; i < num_vars; ++i) {
    Vmath::Neg(num_pts, out_arr[i], 1);
  }

  // Add forcing terms
  for (auto &x : this->fluid_src_terms) {
    x->Apply(m_fields, in_arr, out_arr, time);
  }
}

/**
 * @brief ODE projection method: needs to be defined for explicit time
 * integration.
 */
void SOLSystem::do_ode_projection(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {
  // Do nothing
}

/**
 * @brief Compute the advection terms for the right-hand side
 */
void SOLSystem::do_advection(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time,
    const Array<OneD, const Array<OneD, NekDouble>> &fwd,
    const Array<OneD, const Array<OneD, NekDouble>> &bwd) {
  // Only fields up to and including the energy need to be advected
  int num_fields_to_advect = this->field_to_index.get_idx("E") + 1;

  Array<OneD, Array<OneD, NekDouble>> adv_vel(m_spacedim);
  this->adv_obj->Advect(num_fields_to_advect, m_fields, adv_vel, in_arr,
                        out_arr, time, fwd, bwd);
}

/**
 * @brief Return a flux vector appropriate for the compressible Euler equations.
 *
 * @param fields_vals Physical field values at the quadrature points.
 * @param flux        Resulting flux tensor.
 */
void SOLSystem::get_flux_vector(
    const Array<OneD, const Array<OneD, NekDouble>> &fields_vals,
    TensorOfArray3D<NekDouble> &flux) {
  const auto rho_idx = this->field_to_index.get_idx("rho");
  const auto E_idx = this->field_to_index.get_idx("E");
  // Energy is the last field of relevance, regardless of mesh dimension
  const auto num_vars = E_idx + 1;
  const auto num_pts = fields_vals[0].size();

  // Temporary storage for each point (needed for var converter)
  Array<OneD, NekDouble> field_vals_pt(num_vars);
  Array<OneD, NekDouble> vel_vals_pt(m_spacedim);

  // Point-wise calculation of flux vector
  for (std::size_t pidx = 0; pidx < num_pts; ++pidx) {

    // Extract field vals for this point
    for (std::size_t fidx = 0; fidx < num_vars; ++fidx) {
      field_vals_pt[fidx] = fields_vals[fidx][pidx];
    }

    // 1 / rho
    NekDouble oneOrho = 1.0 / field_vals_pt[rho_idx];

    for (std::size_t dim = 0; dim < m_spacedim; ++dim) {
      // Add momentum densities to flux vector
      flux[0][dim][pidx] = field_vals_pt[dim + 1];
      // Compute velocities from momentum densities
      vel_vals_pt[dim] = field_vals_pt[dim + 1] * oneOrho;
    }

    NekDouble pressure = this->var_converter->GetPressure(field_vals_pt.data());
    NekDouble e_plus_P = field_vals_pt[E_idx] + pressure;
    for (auto dim = 0; dim < m_spacedim; ++dim) {
      // Flux vector for the velocity fields
      for (auto vdim = 0; vdim < m_spacedim; ++vdim) {
        flux[1 + dim][vdim][pidx] = vel_vals_pt[vdim] * field_vals_pt[dim + 1];
      }

      // Add pressure to appropriate field
      flux[1 + dim][dim][pidx] += pressure;

      // Energy flux
      flux[m_spacedim + 1][dim][pidx] = e_plus_P * vel_vals_pt[dim];
    }
  }
}

} // namespace NESO::Solvers::SimpleSOL
