#include "RogersRicci2D.h"

#include <MultiRegions/ContField.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

using namespace Nektar;

namespace NESO::Solvers::H3LAPD {

std::string RogersRicci2D::className =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "RogersRicci2D", RogersRicci2D::create,
        "System for the Rogers-Ricci 2D system of equations.");

class UpwindNeumannSolver : public SU::RiemannSolver {
public:
  UpwindNeumannSolver(const LU::SessionReaderSharedPtr &pSession)
      : SU::RiemannSolver(pSession) {}

  void SetNeumannIdx(std::set<std::size_t> idx) { m_neumannIdx = idx; }

protected:
  std::set<std::size_t> m_neumannIdx;

  void v_Solve(const int nDim,
               const Array<OneD, const Array<OneD, NekDouble>> &Fwd,
               const Array<OneD, const Array<OneD, NekDouble>> &Bwd,
               Array<OneD, Array<OneD, NekDouble>> &flux) final {
    ASSERTL1(CheckScalars("Vn"), "Vn not defined.");
    const Array<OneD, NekDouble> &traceVel = m_scalars["Vn"]();

    for (int j = 0; j < traceVel.size(); ++j) {
      const Array<OneD, const Array<OneD, NekDouble>> &tmp =
          traceVel[j] >= 0 ? Fwd : Bwd;
      for (int i = 0; i < Fwd.size(); ++i) {
        flux[i][j] = traceVel[j] * tmp[i][j];
      }
    }

    // Overwrite Neumann conditions with a zero flux
    for (auto &idx : m_neumannIdx) {
      for (int i = 0; i < Fwd.size(); ++i) {
        flux[i][idx] = 0.0;
      }
    }
  }
};

RogersRicci2D::RogersRicci2D(const LU::SessionReaderSharedPtr &session,
                             const SpatialDomains::MeshGraphSharedPtr &graph)
    : UnsteadySystem(session, graph), AdvectionSystem(session, graph),
      m_driftVel(3) {
  // Set up constants
  /*
  m_c["B"] = c("omega_ci") * c("m_i") * c("q_E");
  m_c["c_s0"] = sqrt(c("T_e0") / c("m_i"));
  m_c["rho_s0"] = c("c_s0") / c("omega_ci");
  m_c["S_0n"] = 0.03 * c("n0") * c("c_s0") / c("R");
  m_c["S_0T"] = 0.03 * c("T_e0") * c("c_s0") / c("R");
  m_c["omega"] = 1.5 * c("R") / c("L_z");
  */
}

void check_var_idx(const LU::SessionReaderSharedPtr session, const int &idx,
                   const std::string var_name) {
  std::stringstream err;
  err << "Expected variable index " << idx << " to correspond to '" << var_name
      << "'. Check your session file.";
  ASSERTL0(session->GetVariable(idx).compare(var_name) == 0, err.str());
}

void check_field_sizes(Array<OneD, MR::ExpListSharedPtr> fields,
                       const int npts) {
  for (auto i = 0; i < fields.size(); i++) {
    ASSERTL0(fields[i]->GetNpoints() == npts,
             "Detected fields with different numbers of quadrature points; "
             "this solver assumes they're all the same");
  }
}

void RogersRicci2D::v_InitObject(bool DeclareField) {
  AdvectionSystem::v_InitObject(DeclareField);

  ASSERTL0(m_fields.size() == 4,
           "Incorrect number of variables detected (expected 4): check your "
           "session file.");

  // Store mesh dimension for easy retrieval later.
  m_ndims = m_graph->GetMeshDimension();
  ASSERTL0(m_ndims == 2 || m_ndims == 3,
           "Solver only supports 2D or 3D meshes.");

  // Check variable order is as expected
  check_var_idx(m_session, n_idx, "n");
  check_var_idx(m_session, Te_idx, "T_e");
  check_var_idx(m_session, w_idx, "w");
  check_var_idx(m_session, phi_idx, "phi");

  // Check fields all have the same number of quad points
  m_npts = m_fields[0]->GetNpoints();
  check_field_sizes(m_fields, m_npts);

  m_fields[phi_idx] = MemoryManager<MR::ContField>::AllocateSharedPtr(
      m_session, m_graph, m_session->GetVariable(phi_idx), true, true);
  m_intVariables = {n_idx, Te_idx, w_idx};

  // Assign storage for drift velocity.
  for (int i = 0; i < m_driftVel.size(); ++i) {
    m_driftVel[i] = Array<OneD, NekDouble>(m_npts, 0.0);
  }

  switch (m_projectionType) {
  case MR::eDiscontinuous: {
    m_homoInitialFwd = false;

    if (m_fields[0]->GetTrace()) {
      m_traceVn = Array<OneD, NekDouble>(GetTraceNpoints());
    }

    std::string advName, riemName;
    m_session->LoadSolverInfo("AdvectionType", advName, "WeakDG");
    m_advObject = SU::GetAdvectionFactory().CreateInstance(advName, advName);
    m_advObject->SetFluxVector(&RogersRicci2D::GetFluxVector, this);

    m_riemannSolver = std::make_shared<UpwindNeumannSolver>(m_session);
    m_riemannSolver->SetScalar("Vn", &RogersRicci2D::GetNormalVelocity, this);

    // Tell the advection object about the Riemann solver to use, and
    // then get it set up.
    m_advObject->SetRiemannSolver(m_riemannSolver);
    m_advObject->InitObject(m_session, m_fields);
    break;
  }

  default: {
    ASSERTL0(false, "Unsupported projection type: only discontinuous"
                    " projection supported.");
    break;
  }
  }

  m_ode.DefineOdeRhs(&RogersRicci2D::ExplicitTimeInt, this);
  m_ode.DefineProjection(&RogersRicci2D::DoOdeProjection, this);

  if (!m_explicitAdvection) {
    m_implHelper = std::make_shared<ImplicitHelper>(m_session, m_fields, m_ode,
                                                    m_intVariables.size());
    m_implHelper->InitialiseNonlinSysSolver();
    m_ode.DefineImplicitSolve(&ImplicitHelper::ImplicitTimeInt, m_implHelper);
  }

  // Store distance of quad points from origin in transverse plane.
  // (used to compute source terms)
  Array<OneD, NekDouble> x = Array<OneD, NekDouble>(m_npts);
  Array<OneD, NekDouble> y = Array<OneD, NekDouble>(m_npts);
  m_r = Array<OneD, NekDouble>(m_npts);
  if (m_ndims == 3) {
    Array<OneD, NekDouble> z = Array<OneD, NekDouble>(m_npts);
    m_fields[0]->GetCoords(x, y, z);
  } else {
    m_fields[0]->GetCoords(x, y);
  }
  for (int i = 0; i < m_npts; ++i) {
    m_r[i] = sqrt(x[i] * x[i] + y[i] * y[i]);
  }

  // Figure out Neumann quadrature in trace
  const Array<OneD, const int> &traceBndMap = m_fields[0]->GetTraceBndMap();
  std::set<std::size_t> neumannIdx;
  for (size_t n = 0, cnt = 0;
       n < (size_t)m_fields[0]->GetBndConditions().size(); ++n) {
    if (m_fields[0]->GetBndConditions()[n]->GetBoundaryConditionType() ==
        SpatialDomains::ePeriodic) {
      continue;
    }

    int nExp = m_fields[0]->GetBndCondExpansions()[n]->GetExpSize();

    if (m_fields[0]->GetBndConditions()[n]->GetBoundaryConditionType() !=
        SpatialDomains::eNeumann) {
      cnt += nExp;
      continue;
    }

    for (int e = 0; e < nExp; ++e) {
      auto nBCEdgePts =
          m_fields[0]->GetBndCondExpansions()[n]->GetExp(e)->GetTotPoints();

      auto id = m_fields[0]->GetTrace()->GetPhys_Offset(traceBndMap[cnt + e]);
      for (int q = 0; q < nBCEdgePts; ++q) {
        neumannIdx.insert(id + q);
      }
    }

    cnt += nExp;
  }

  std::dynamic_pointer_cast<UpwindNeumannSolver>(m_riemannSolver)
      ->SetNeumannIdx(neumannIdx);
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
 * - First, compute the electrostatic potential \f$ \phi \f$, given the
 * - Using this, compute the drift velocity \f$ (\partial_y\phi,
 *   -\partial_x\phi).
 * - Then evaluate the \f$ \nabla\cdot\mathbf{F} \f$ operator using the
 *   advection object #m_advObject.
 * - Finally put this on the right hand side and evaluate the source terms for
 *   each field.
 *
 * The assumption here is that fields are ordered inside `m_fields` so that
 * field 0 is vorticity \f$ \zeta \f$, field 1 is number density \f$ n \f$, and
 * field 2 is electrostatic potential. Only \f$ \zeta \f$ and \f$ n \f$ are time
 * integrated.
 *
 * @param inarray    Array containing each field's current state.
 * @param outarray   The result of the right-hand side operator for each field
 *                   being time integrated.
 * @param time       Current value of time.
 */
void RogersRicci2D::ExplicitTimeInt(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {
  // nPts below corresponds to the total number of solution/integration
  // points: i.e. number of elements * quadrature points per element.
  int i;

  // Set up factors for electrostatic potential solve. We support a generic
  // Helmholtz solve of the form (\nabla^2 - \lambda) u = f, so this sets
  // \lambda to zero.
  StdRegions::ConstFactorMap factors;
  factors[StdRegions::eFactorLambda] = 0.0;
  factors[StdRegions::eFactorCoeffD22] = 0.0;

  Vmath::Zero(m_fields[phi_idx]->GetNcoeffs(),
              m_fields[phi_idx]->UpdateCoeffs(), 1);

  // Solve for phi. Output of this routine is in coefficient (spectral) space,
  // so backwards transform to physical space since we'll need that for the
  // advection step & computing drift velocity.
  m_fields[phi_idx]->HelmSolve(inarray[w_idx],
                               m_fields[phi_idx]->UpdateCoeffs(), factors);
  m_fields[phi_idx]->BwdTrans(m_fields[phi_idx]->GetCoeffs(),
                              m_fields[phi_idx]->UpdatePhys());

  // Calculate drift velocity v_E: PhysDeriv takes input and computes spatial
  // derivatives.
  Array<OneD, NekDouble> dummy = Array<OneD, NekDouble>(m_npts);
  m_fields[phi_idx]->PhysDeriv(m_fields[phi_idx]->GetPhys(), m_driftVel[1],
                               m_driftVel[0], dummy);

  // We frequently use vector math (Vmath) routines for one-line operations
  // like negating entries in a vector.
  Vmath::Neg(m_npts, m_driftVel[1], 1);

  // Do advection for zeta, n. The hard-coded '3' here indicates that we
  // should only advect the first two components of inarray.
  m_advObject->Advect(3, m_fields, m_driftVel, inarray, outarray, time);

  Array<OneD, NekDouble> n = inarray[n_idx];
  Array<OneD, NekDouble> T_e = inarray[Te_idx];
  Array<OneD, NekDouble> w = inarray[w_idx];
  Array<OneD, NekDouble> phi = m_fields[phi_idx]->UpdatePhys();

  // Put advection term on the right hand side.
  const NekDouble rho_s0 = 1.2e-2;
  const NekDouble r_s = 20 * rho_s0; // can't find this in list of constants,
                                     // stolen from rr.py... fingers crossed

  // stolen from Ed/Owen's code, rr.py
  const NekDouble Ls_boost = 2.0;
  const NekDouble L_s = 0.5 * rho_s0 * Ls_boost; // maybe wrong

  for (i = 0; i < m_npts; ++i) {
    NekDouble et = exp(3 - phi[i] / sqrt(T_e[i] * T_e[i] + 1e-4));
    NekDouble st = 0.03 * (1.0 - tanh((rho_s0 * m_r[i] - r_s) / L_s));
    outarray[n_idx][i] = -40 * outarray[n_idx][i] - 1.0 / 24.0 * et * n[i] + st;
    outarray[Te_idx][i] = -40 * outarray[Te_idx][i] -
                          1.0 / 36.0 * (1.71 * et - 0.71) * T_e[i] + st;
    outarray[w_idx][i] = -40 * outarray[w_idx][i] + 1.0 / 24.0 * (1 - et);
  }
}

/**
 * @brief Perform projection into correct polynomial space.
 *
 * This routine projects the @p inarray input and ensures the @p outarray output
 * lives in the correct space. Since we are hard-coding DG, this corresponds to
 * a simple copy from in to out, since no elemental connectivity is required and
 * the output of the RHS function is polynomial.
 */
void RogersRicci2D::DoOdeProjection(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {
  int nvariables = inarray.size(), npoints = GetNpoints();
  SetBoundaryConditions(time);

  for (int i = 0; i < nvariables; ++i) {
    Vmath::Vcopy(npoints, inarray[i], 1, outarray[i], 1);
  }
}

/**
 * @brief Compute the flux vector for this system.
 */
void RogersRicci2D::GetFluxVector(
    const Array<OneD, Array<OneD, NekDouble>> &physfield,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux) {
  ASSERTL1(flux[0].size() == m_driftVel.size(),
           "Dimension of flux array and velocity array do not match");

  int nq = physfield[0].size();

  for (int i = 0; i < flux.size(); ++i) {
    for (int j = 0; j < flux[0].size(); ++j) {
      Vmath::Vmul(nq, physfield[i], 1, m_driftVel[j], 1, flux[i][j], 1);
    }
  }
}

/**
 * @brief Compute the normal advection velocity for this system on the
 * trace/skeleton/edges of the 2D mesh.
 */
Array<OneD, NekDouble> &RogersRicci2D::GetNormalVelocity() {
  // Number of trace (interface) points
  int nTracePts = GetTraceNpoints();

  // Auxiliary variable to compute the normal velocity
  Array<OneD, NekDouble> tmp(nTracePts);

  // Reset the normal velocity
  Vmath::Zero(nTracePts, m_traceVn, 1);

  // Compute dot product of velocity along trace with trace normals. Store in
  // m_traceVn.
  for (int i = 0; i < m_ndims; ++i) {
    m_fields[0]->ExtractTracePhys(m_driftVel[i], tmp);

    Vmath::Vvtvp(nTracePts, m_traceNormals[i], 1, tmp, 1, m_traceVn, 1,
                 m_traceVn, 1);
  }

  return m_traceVn;
}

} // namespace NESO::Solvers::H3LAPD
