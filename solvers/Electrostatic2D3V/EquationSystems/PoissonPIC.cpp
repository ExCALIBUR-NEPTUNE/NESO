#include "PoissonPIC.hpp"

namespace NESO::Solvers::Electrostatic2D3V {
std::string PoissonPIC::className1 =
    SU::GetEquationSystemFactory().RegisterCreatorFunction("PoissonPIC",
                                                           PoissonPIC::create);
std::string PoissonPIC::className2 =
    SU::GetEquationSystemFactory().RegisterCreatorFunction("SteadyDiffusion",
                                                           PoissonPIC::create);

PoissonPIC::PoissonPIC(const LU::SessionReaderSharedPtr &session,
                       const SD::MeshGraphSharedPtr &graph)
    : SU::EquationSystem(session, graph), m_factors() {
  m_factors[SR::eFactorLambda] = 0.0;
  m_factors[SR::eFactorTau] = 1.0;
  auto variables = session->GetVariables();
  int index = 0;
  for (auto vx : variables) {
    this->field_to_index[vx] = index;
    index++;
  }
  ASSERTL1(index == 2,
           "Expected to read 2 fields from session context: u and rho.");
  ASSERTL1(this->GetFieldIndex("u") > -1, "Could not get index for u.");
  ASSERTL1(this->GetFieldIndex("rho") > -1, "Could not get index for rho.");
}

int PoissonPIC::GetFieldIndex(const std::string name) {
  ASSERTL1(this->field_to_index.count(name) > 0,
           "Could not map field name to index.");
  return this->field_to_index[name];
}

void PoissonPIC::v_InitObject(bool DeclareFields) {
  SU::EquationSystem::v_InitObject(true);
}

PoissonPIC::~PoissonPIC() {}

void PoissonPIC::v_GenerateSummary(SU::SummaryList &s) {
  SU::EquationSystem::SessionSummary(s);
}

Nektar::Array<Nektar::OneD, bool> PoissonPIC::v_GetSystemSingularChecks() {
  auto singular_bools = Nektar::Array<Nektar::OneD, bool>(
      m_session->GetVariables().size(), false);
  singular_bools[this->GetFieldIndex("u")] = true;
  return singular_bools;
}

void PoissonPIC::v_DoSolve() {
  const int u_index = this->GetFieldIndex("u");
  const int rho_index = this->GetFieldIndex("rho");
  Vmath::Zero(m_fields[u_index]->GetNcoeffs(),
              m_fields[u_index]->UpdateCoeffs(), 1);
  m_fields[u_index]->HelmSolve(m_fields[rho_index]->GetPhys(),
                               m_fields[u_index]->UpdateCoeffs(), m_factors);
  Vmath::Zero(m_fields[u_index]->GetTotPoints(),
              m_fields[u_index]->UpdatePhys(), 1);
  m_fields[u_index]->BwdTrans(m_fields[u_index]->UpdateCoeffs(),
                              m_fields[u_index]->UpdatePhys());
  m_fields[u_index]->SetPhysState(true);
}

} // namespace NESO::Solvers::Electrostatic2D3V
