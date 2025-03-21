#include "CwipiUtils.hpp"

#include "CwipiDiffTensorSender.hpp"

namespace NESO::Solvers::Diffusion {
std::string CwipiDiffTensorSender::className =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "CWIPI_DiffTensorSender", CwipiDiffTensorSender::create);

CwipiDiffTensorSender::CwipiDiffTensorSender(
    const LU::SessionReaderSharedPtr &session,
    const SD::MeshGraphSharedPtr &graph)
    : DiffusionSystem(session, graph) {}

void CwipiDiffTensorSender::do_nothing(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time,
    const NekDouble lambda) {
  boost::ignore_unused(in_arr, out_arr, time, lambda);
}

/**
 * @brief Initialise coupling object, bind a dummy solver function.
 */
void CwipiDiffTensorSender::v_InitObject(bool DeclareField) {
  // Parent func reads params and sets diffusion tensor coeffs ready to send
  DiffusionSystem::v_InitObject(DeclareField);

  this->coupling = construct_coupling_obj(m_session, m_fields[0]);

  // This eqn sys doesn't evolve fields - bind a dummy solve function
  m_ode.DefineImplicitSolve(&CwipiDiffTensorSender::do_nothing, this);
}

bool CwipiDiffTensorSender::v_PreIntegrate(int step) {
  // Send diff tensor coeffs to the other cwipi-enabled exec (1st step only)
  if (this->coupling && step == 0) {
    // Extract coeff values from map into a Nektar Array
    std::vector<std::string> fld_names = {"d00", "d01", "d11"};
    Nektar::Array<Nektar::OneD, Nektar::Array<Nektar::OneD, Nektar::NekDouble>>
        send_arr(this->helmsolve_varcoeffs.size());
    send_arr[0] =
        this->helmsolve_varcoeffs[StdRegions::eVarCoeffD00].GetValue();
    send_arr[1] =
        this->helmsolve_varcoeffs[StdRegions::eVarCoeffD01].GetValue();
    send_arr[2] =
        this->helmsolve_varcoeffs[StdRegions::eVarCoeffD11].GetValue();

    // Send array
    this->coupling->Send(step, m_time, send_arr, fld_names);
  }

  return DiffusionSystem::v_PreIntegrate(step);
}

/**
 * @brief Unsteady diffusion problem destructor.
 */
CwipiDiffTensorSender::~CwipiDiffTensorSender() {}

void CwipiDiffTensorSender::v_GenerateSummary(SU::SummaryList &s) {
  DiffusionSystem::v_GenerateSummary(s);
  // Other
}

} // namespace NESO::Solvers::Diffusion
