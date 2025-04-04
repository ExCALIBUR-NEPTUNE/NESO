#include "CwipiUtils.hpp"

#include "CwipiReceiveDiffTensorAndDiffuse.hpp"

namespace NESO::Solvers::Diffusion {
std::string CwipiReceiveDiffTensorAndDiffuse::className =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "CWIPI_ReceiveDiffTensorAndDiffuse",
        CwipiReceiveDiffTensorAndDiffuse::create);

CwipiReceiveDiffTensorAndDiffuse::CwipiReceiveDiffTensorAndDiffuse(
    const LU::SessionReaderSharedPtr &session,
    const SD::MeshGraphSharedPtr &graph)
    : DiffusionSystem(session, graph) {}

/**
 * @brief Initialisation object for the unsteady diffusion problem.
 */
void CwipiReceiveDiffTensorAndDiffuse::v_InitObject(bool DeclareField) {
  TimeEvoEqnSysBase::v_InitObject(DeclareField);

  this->coupling = construct_coupling_obj(m_session, m_fields[0]);

  // CG only
  ASSERTL0(m_projectionType == MR::eGalerkin,
           "Only continuous Galerkin discretisation supported.");

  // Re-use parent class solve function
  m_ode.DefineImplicitSolve(
      &CwipiReceiveDiffTensorAndDiffuse::do_implicit_solve, this);
}

bool CwipiReceiveDiffTensorAndDiffuse::v_PreIntegrate(int step) {
  // Receive difftensor coeffs from the other cwipi-enabled exec (1st step only)
  if (this->coupling && step == 0) {
    std::vector<std::string> fld_names = {"d00", "d01", "d11"};
    Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, Nektar::NekDouble>>
        rcv_arr(fld_names.size());
    for (auto ii = 0; ii < fld_names.size(); ii++) {
      rcv_arr[ii] = Array<OneD, NekDouble>(this->n_pts, 0.0);
    }

    // Receive array
    this->coupling->Receive(step, m_time, rcv_arr, fld_names);

    //===== Special behaviour for MatrixFree collections =====
    std::stringstream ss;
    switch (get_collection_type()) {
    case NC::ImplementationType::eMatrixFree:
      // For matrix-free collections set (first element of) received values as
      // constant coefficients
      this->helmsolve_factors[SR::eFactorCoeffD00] = rcv_arr[0][0];
      this->helmsolve_factors[SR::eFactorCoeffD01] = rcv_arr[1][0];
      this->helmsolve_factors[SR::eFactorCoeffD11] = rcv_arr[2][0];
      break;
    default:
      // For all other implementations, or if no collection was specified, copy
      // received values into a VarCoeffMap for helmsolve
      this->helmsolve_varcoeffs[StdRegions::eVarCoeffD00] = rcv_arr[0];
      this->helmsolve_varcoeffs[StdRegions::eVarCoeffD01] = rcv_arr[1];
      this->helmsolve_varcoeffs[StdRegions::eVarCoeffD11] = rcv_arr[2];
      break;
    }
    //===== End of special behaviour for MatrixFree collections =====
  }

  return DiffusionSystem::v_PreIntegrate(step);
}

/**
 * @brief Unsteady diffusion problem destructor.
 */
CwipiReceiveDiffTensorAndDiffuse::~CwipiReceiveDiffTensorAndDiffuse() {}

void CwipiReceiveDiffTensorAndDiffuse::v_GenerateSummary(SU::SummaryList &s) {
  DiffusionSystem::v_GenerateSummary(s);
  // Other
}

} // namespace NESO::Solvers::Diffusion
