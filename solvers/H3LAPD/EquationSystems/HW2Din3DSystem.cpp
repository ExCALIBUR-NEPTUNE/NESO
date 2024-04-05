#include "HW2Din3DSystem.hpp"
#include "neso_particles.hpp"
#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <boost/core/ignore_unused.hpp>

namespace NESO::Solvers::H3LAPD {
std::string HW2Din3DSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "2Din3DHW", HW2Din3DSystem::create,
        "(2D) Hasegawa-Wakatani equation system as an intermediate step "
        "towards the full H3-LAPD problem");

HW2Din3DSystem::HW2Din3DSystem(const LU::SessionReaderSharedPtr &session,
                               const SD::MeshGraphSharedPtr &graph)
    : UnsteadySystem(session, graph), AdvectionSystem(session, graph),
      DriftReducedSystem(session, graph), HWSystem(session, graph) {}

/**
 * @brief Populate rhs array ( @p out_arr ) for explicit time integration of
 * the 2D Hasegawa Wakatani equations.
 *
 * @param in_arr physical values of all fields
 * @param[out] out_arr output array (RHSs of time integration equations)
 */
void HW2Din3DSystem::explicit_time_int(
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

  // Add \alpha*(\phi-n_e) to RHS
  Array<OneD, NekDouble> HWterm_2D_alpha(npts);
  Vmath::Vsub(npts, m_fields[phi_idx]->GetPhys(), 1,
              m_fields[ne_idx]->GetPhys(), 1, HWterm_2D_alpha, 1);
  Vmath::Smul(npts, m_alpha, HWterm_2D_alpha, 1, HWterm_2D_alpha, 1);
  Vmath::Vadd(npts, out_arr[w_idx], 1, HWterm_2D_alpha, 1, out_arr[w_idx], 1);
  Vmath::Vadd(npts, out_arr[ne_idx], 1, HWterm_2D_alpha, 1, out_arr[ne_idx], 1);

  // Add \kappa*\dpartial\phi/\dpartial y to RHS
  Array<OneD, NekDouble> HWterm_2D_kappa(npts);
  m_fields[phi_idx]->PhysDeriv(1, m_fields[phi_idx]->GetPhys(),
                               HWterm_2D_kappa);
  Vmath::Smul(npts, m_kappa, HWterm_2D_kappa, 1, HWterm_2D_kappa, 1);
  Vmath::Vsub(npts, out_arr[ne_idx], 1, HWterm_2D_kappa, 1, out_arr[ne_idx], 1);

  // Add particle sources
  add_particle_sources({"ne"}, out_arr);
}

/**
 * @brief Read base class params then extra params required for 2D-in-3D HW.
 */
void HW2Din3DSystem::load_params() {
  DriftReducedSystem::load_params();

  // alpha (required)
  m_session->LoadParameter("HW_alpha", m_alpha);

  // kappa (required)
  m_session->LoadParameter("HW_kappa", m_kappa);
}

void HW2Din3DSystem::init_nonlinsys_solver() {
  int ntotal = 2 * m_fields[0]->GetNpoints();

  // Create the key to hold settings for nonlin solver
  LU::NekSysKey key = LU::NekSysKey();

  // Load required LinSys parameters:
  m_session->LoadParameter("NekLinSysMaxIterations",
                           key.m_NekLinSysMaxIterations, 30);
  m_session->LoadParameter("LinSysMaxStorage", key.m_LinSysMaxStorage, 30);
  m_session->LoadParameter("LinSysRelativeTolInNonlin",
                           key.m_NekLinSysTolerance, 5.0E-2);
  m_session->LoadParameter("GMRESMaxHessMatBand", key.m_KrylovMaxHessMatBand,
                           31);

  // Load required NonLinSys parameters:
  m_session->LoadParameter("JacobiFreeEps", m_jacobiFreeEps, 5.0E-8);
  m_session->LoadParameter("NekNonlinSysMaxIterations",
                           key.m_NekNonlinSysMaxIterations, 10);
  // m_session->LoadParameter("NewtonRelativeIteTol",
  //                         key.m_NekNonLinSysTolerance, 1.0E-12);
  WARNINGL0(!m_session->DefinesParameter("NewtonAbsoluteIteTol"),
            "Please specify NewtonRelativeIteTol instead of "
            "NewtonAbsoluteIteTol in XML session file");
  m_session->LoadParameter("NonlinIterTolRelativeL2",
                           key.m_NonlinIterTolRelativeL2, 1.0E-3);
  m_session->LoadSolverInfo("LinSysIterSolverTypeInNonlin",
                            key.m_LinSysIterSolverTypeInNonlin, "GMRES");

  LU::NekSysOperators nekSysOp;
  nekSysOp.DefineNekSysResEval(&HW2Din3DSystem::nonlinsys_evaluator_1D, this);
  nekSysOp.DefineNekSysLhsEval(&HW2Din3DSystem::matrix_multiply_MF, this);
  nekSysOp.DefineNekSysPrecon(&HW2Din3DSystem::do_null_precon, this);

  // Initialize non-linear system
  m_nonlinsol = LU::GetNekNonlinSysIterFactory().CreateInstance(
      "Newton", m_session, m_comm->GetRowComm(), ntotal, key);
  m_nonlinsol->SetSysOperators(nekSysOp);
}

void HW2Din3DSystem::implicit_time_int(
    const Array<OneD, const Array<OneD, NekDouble>> &inpnts,
    Array<OneD, Array<OneD, NekDouble>> &outpnt, const NekDouble time,
    const NekDouble lambda) {
  unsigned int nvariables = inpnts.size();
  unsigned int npoints = m_fields[0]->GetNpoints();
  unsigned int ntotal = nvariables * npoints;

  Array<OneD, NekDouble> inarray(ntotal);
  Array<OneD, NekDouble> outarray(ntotal);

  for (int i = 0; i < nvariables; ++i) {
    int noffset = i * npoints;
    Array<OneD, NekDouble> tmp;
    Vmath::Vcopy(npoints, inpnts[i], 1, tmp = inarray + noffset, 1);
  }

  implicit_time_int_1D(inarray, outarray, time, lambda);

  for (int i = 0; i < nvariables; ++i) {
    int noffset = i * npoints;
    Vmath::Vcopy(npoints, outarray + noffset, 1, outpnt[i], 1);
  }
}

void HW2Din3DSystem::implicit_time_int_1D(
    const Array<OneD, const NekDouble> &inarray, Array<OneD, NekDouble> &out,
    const NekDouble time, const NekDouble lambda) {
  m_TimeIntegLambda = lambda;
  m_bndEvaluateTime = time;
  unsigned int ntotal = inarray.size();

  if (m_inArrayNorm < 0.0) {
    calc_ref_values(inarray);
  }

  // m_nonlinsol->SetRhsMagnitude(m_inArrayNorm);

  m_tot_newton_its += m_nonlinsol->SolveSystem(ntotal, inarray, out, 0);
  m_tot_lin_its += m_nonlinsol->GetNtotLinSysIts();

  m_TotImpStages++;
}

void HW2Din3DSystem::calc_ref_values(
    const Array<OneD, const NekDouble> &inarray) {
  unsigned int nvariables = m_fields.size();
  unsigned int ntotal = inarray.size();
  unsigned int npoints = ntotal / nvariables;

  Array<OneD, NekDouble> magnitdEstimat(2, 0.0);

  for (int i = 0; i < 2; ++i) {
    int offset = i * npoints;
    magnitdEstimat[i] = Vmath::Dot(npoints, inarray + offset, inarray + offset);
  }
  m_comm->GetSpaceComm()->AllReduce(magnitdEstimat,
                                    Nektar::LibUtilities::ReduceSum);

  m_inArrayNorm = 0.0;
  for (int i = 0; i < 2; ++i) {
    m_inArrayNorm += magnitdEstimat[i];
  }
}

void HW2Din3DSystem::nonlinsys_evaluator_1D(
    const Array<OneD, const NekDouble> &inarray, Array<OneD, NekDouble> &out,
    [[maybe_unused]] const bool &flag) {
  unsigned int npoints = m_fields[0]->GetNpoints();
  Array<OneD, Array<OneD, NekDouble>> in2D(2);
  Array<OneD, Array<OneD, NekDouble>> out2D(2);
  for (int i = 0; i < 2; ++i) {
    int offset = i * npoints;
    in2D[i] = inarray + offset;
    out2D[i] = out + offset;
  }
  nonlinsys_evaluator(in2D, out2D);
}

void HW2Din3DSystem::nonlinsys_evaluator(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &out) {
  unsigned int npoints = m_fields[0]->GetNpoints();
  Array<OneD, Array<OneD, NekDouble>> inpnts(2);
  for (int i = 0; i < 2; ++i) {
    inpnts[i] = Array<OneD, NekDouble>(npoints, 0.0);
  }

  do_ode_projection(inarray, inpnts, m_bndEvaluateTime);
  explicit_time_int(inpnts, out, m_bndEvaluateTime);

  for (int i = 0; i < 2; ++i) {
    Vmath::Svtvp(npoints, -m_TimeIntegLambda, out[i], 1, inarray[i], 1, out[i],
                 1);
    Vmath::Vsub(npoints, out[i], 1,
                m_nonlinsol->GetRefSourceVec() + i * npoints, 1, out[i], 1);
  }
}

void HW2Din3DSystem::matrix_multiply_MF(
    const Array<OneD, const NekDouble> &inarray, Array<OneD, NekDouble> &out,
    [[maybe_unused]] const bool &flag) {
  const Array<OneD, const NekDouble> solref = m_nonlinsol->GetRefSolution();
  const Array<OneD, const NekDouble> resref = m_nonlinsol->GetRefResidual();

  unsigned int ntotal = inarray.size();
  NekDouble magninarray = Vmath::Dot(ntotal, inarray, inarray);
  m_comm->GetSpaceComm()->AllReduce(magninarray,
                                    Nektar::LibUtilities::ReduceSum);
  NekDouble eps =
      m_jacobiFreeEps * sqrt((sqrt(m_inArrayNorm) + 1.0) / magninarray);

  Array<OneD, NekDouble> solplus{ntotal};
  Array<OneD, NekDouble> resplus{ntotal};

  Vmath::Svtvp(ntotal, eps, inarray, 1, solref, 1, solplus, 1);
  nonlinsys_evaluator_1D(solplus, resplus, flag);
  Vmath::Vsub(ntotal, resplus, 1, resref, 1, out, 1);
  Vmath::Smul(ntotal, 1.0 / eps, out, 1, out, 1);
}

void HW2Din3DSystem::do_null_precon(const Array<OneD, NekDouble> &inarray,
                                    Array<OneD, NekDouble> &outarray,
                                    [[maybe_unused]] const bool &flag) {
  Vmath::Vcopy(inarray.size(), inarray, 1, outarray, 1);
}

/**
 * @brief Post-construction class-initialisation.
 */
void HW2Din3DSystem::v_InitObject(bool DeclareField) {
  HWSystem::v_InitObject(DeclareField);

  // Bind RHS function for implicit time integration
  m_ode.DefineImplicitSolve(&HW2Din3DSystem::implicit_time_int, this);
  // Bind RHS function for explicit time integration
  m_ode.DefineOdeRhs(&HW2Din3DSystem::explicit_time_int, this);

  if (!m_explicitAdvection) {
    init_nonlinsys_solver();
  }

  // Create diagnostic for recording growth rates
  if (m_diag_growth_rates_recording_enabled) {
    m_diag_growth_rates_recorder =
        std::make_shared<GrowthRatesRecorder<MultiRegions::DisContField>>(
            m_session, 2, m_discont_fields["ne"], m_discont_fields["w"],
            m_discont_fields["phi"], GetNpoints(), m_alpha, m_kappa);
  }
}

} // namespace NESO::Solvers::H3LAPD
