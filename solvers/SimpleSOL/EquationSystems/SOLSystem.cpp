#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

#include "SOLSystem.hpp"

namespace NESO::Solvers {
std::string SOLSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "SOL", SOLSystem::create, "SOL equations in conservative variables.");

SOLSystem::SOLSystem(const LU::SessionReaderSharedPtr &session,
                     const SD::MeshGraphSharedPtr &graph)
    : UnsteadySystem(session, graph),
      m_field_to_index(session->GetVariables()) {

  // m_spacedim isn't set at this point, for some reason; use mesh dim instead
  NESOASSERT(graph->GetSpaceDimension() == 1 || graph->GetSpaceDimension() == 2,
             "Unsupported mush dimension for SOLSystem - must be 1 or 2.");
  if (graph->GetSpaceDimension() == 2) {
    m_required_flds = {"rho", "rhou", "rhov", "E"};
  } else {
    m_required_flds = {"rho", "rhou", "E"};
  }
  m_int_fld_names = std::vector<std::string>(m_required_flds);
}

/**
 * Check all required fields are defined
 */
void SOLSystem::validate_field_list() {
  for (auto &fld_name : m_required_flds) {
    ASSERTL0(m_field_to_index.get_idx(fld_name) >= 0,
             "Required field [" + fld_name + "] is not defined.");
  }
}

/**
 * @brief Initialization object for SOLSystem class.
 */
void SOLSystem::v_InitObject(bool DeclareField) {
  validate_field_list();
  UnsteadySystem::v_InitObject(DeclareField);

  // Tell UnsteadySystem to only integrate a subset of fields in time
  // (Ignore fields that don't have a time derivative)
  m_intVariables.resize(m_int_fld_names.size());
  for (auto ii = 0; ii < m_int_fld_names.size(); ii++) {
    int var_idx = m_field_to_index.get_idx(m_int_fld_names[ii]);
    ASSERTL0(var_idx >= 0, "Setting time integration vars - GetIntFieldNames() "
                           "returned an invalid field name.");
    m_intVariables[ii] = var_idx;
  }

  for (int i = 0; i < m_fields.size(); i++) {
    // Use BwdTrans to make sure initial condition is in solution space
    m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(), m_fields[i]->UpdatePhys());
  }

  m_var_converter = MemoryManager<VariableConverter>::AllocateSharedPtr(
      m_session, m_spacedim);

  ASSERTL0(m_session->DefinesSolverInfo("UPWINDTYPE"),
           "No UPWINDTYPE defined in session.");

  // Store velocity field indices for the Riemann solver.
  m_vec_locs = Array<OneD, Array<OneD, NekDouble>>(1);
  m_vec_locs[0] = Array<OneD, NekDouble>(m_spacedim);
  for (int i = 0; i < m_spacedim; ++i) {
    m_vec_locs[0][i] = 1 + i;
  }

  // Loading parameters from session file
  m_session->LoadParameter("Gamma", m_gamma, 1.4);

  // Setting up advection and diffusion operators
  init_advection();

  // Set up forcing/source term objects.
  m_forcing = SU::Forcing::Load(m_session, shared_from_this(), m_fields,
                                m_fields.size());

  m_ode.DefineOdeRhs(&SOLSystem::explicit_time_int, this);
  m_ode.DefineProjection(&SOLSystem::do_ode_projection, this);
}

/**
 * @brief Destructor for SOLSystem class.
 */
SOLSystem::~SOLSystem() {}

/**
 * @brief Initialisation, including creation of advection object.
 */
void SOLSystem::init_advection() {
  // Only DG is supported
  ASSERTL0(m_projectionType == MR::eDiscontinuous,
           "Unsupported projection type: must be DG.");

  std::string adv_type, riemann_type;
  m_session->LoadSolverInfo("AdvectionType", adv_type, "WeakDG");

  m_adv = SU::GetAdvectionFactory().CreateInstance(adv_type, adv_type);

  m_adv->SetFluxVector(&SOLSystem::get_flux_vector, this);

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
  m_adv->SetRiemannSolver(riemann_solver);
  m_adv->InitObject(m_session, m_fields);
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
  for (auto &x : m_forcing) {
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
  int num_fields_to_advect = m_field_to_index.get_idx("E") + 1;

  Array<OneD, Array<OneD, NekDouble>> adv_vel(m_spacedim);
  m_adv->Advect(num_fields_to_advect, m_fields, adv_vel, in_arr, out_arr, time,
                fwd, bwd);
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
  // Energy is the last field of relevance, regardless of mesh dimension
  const auto E_idx = m_field_to_index.get_idx("E");
  const auto num_vars = E_idx + 1;
  const auto num_pts = fields_vals[0].size();

  // Temporary space for 2 velocity fields (second one is ignored in 1D)
  constexpr unsigned short num_all_flds = 4;
  constexpr unsigned short num_vel_flds = 2;

  for (std::size_t p = 0; p < num_pts; ++p) {
    // Create local storage
    std::array<NekDouble, num_all_flds> all_phys;
    std::array<NekDouble, num_vel_flds> vel_phys;

    // Copy phys vals for this point
    for (std::size_t f = 0; f < num_vars; ++f) {
      all_phys[f] = fields_vals[f][p];
    }

    // 1 / rho
    NekDouble oneOrho = 1.0 / all_phys[0];

    for (std::size_t dim = 0; dim < m_spacedim; ++dim) {
      // Add momentum densities to flux vector
      flux[0][dim][p] = all_phys[dim + 1];
      // Compute velocities from momentum densities
      vel_phys[dim] = all_phys[dim + 1] * oneOrho;
    }

    NekDouble pressure = m_var_converter->GetPressure(all_phys.data());
    NekDouble e_plus_P = all_phys[E_idx] + pressure;
    for (auto dim = 0; dim < m_spacedim; ++dim) {
      // Flux vector for the velocity fields
      for (auto vdim = 0; vdim < m_spacedim; ++vdim) {
        flux[dim + 1][vdim][p] = vel_phys[vdim] * all_phys[dim + 1];
      }

      // Add pressure to appropriate field
      flux[dim + 1][dim][p] += pressure;

      // Energy flux
      flux[m_spacedim + 1][dim][p] = e_plus_P * vel_phys[dim];
    }
  }
}

} // namespace NESO::Solvers
