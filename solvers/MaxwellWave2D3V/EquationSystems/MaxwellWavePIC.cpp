#include "MaxwellWavePIC.h"

using namespace std;

namespace Nektar {
string MaxwellWavePIC::className1 =
    GetEquationSystemFactory().RegisterCreatorFunction("MaxwellWavePIC",
                                                       MaxwellWavePIC::create);
string MaxwellWavePIC::className2 =
    GetEquationSystemFactory().RegisterCreatorFunction("SteadyDiffusion",
                                                       MaxwellWavePIC::create);

MaxwellWavePIC::MaxwellWavePIC(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph), m_factors() {
  m_factors[StdRegions::eFactorLambda] = 0.0;
  m_factors[StdRegions::eFactorTau] = 1.0;
  auto variables = pSession->GetVariables();
  int index = 0;
  for (auto vx : variables) {
    this->field_to_index[vx] = index;
    index++;
  }
  // ASSERTL1(index == 2,
  //          "Expected to read 2 fields from session context: u and rho.");
  //  suffixed underscores used for variables at the last timestep
  ASSERTL1(this->GetFieldIndex("rho") > -1, "Could not get index for rho.");
  ASSERTL1(this->GetFieldIndex("Jx") > -1, "Could not get index for Jx.");
  ASSERTL1(this->GetFieldIndex("Jy") > -1, "Could not get index for Jy.");
  ASSERTL1(this->GetFieldIndex("Jz") > -1, "Could not get index for Jz.");
  ASSERTL1(this->GetFieldIndex("phi") > -1, "Could not get index for phi.");
  ASSERTL1(this->GetFieldIndex("phi_minus") > -1,
           "Could not get index for phi_minus.");
  ASSERTL1(this->GetFieldIndex("Ax") > -1, "Could not get index for Ax.");
  ASSERTL1(this->GetFieldIndex("Ay") > -1, "Could not get index for Ay.");
  ASSERTL1(this->GetFieldIndex("Az") > -1, "Could not get index for Az.");
  ASSERTL1(this->GetFieldIndex("Ax_minus") > -1,
           "Could not get index for Ax_minus.");
  ASSERTL1(this->GetFieldIndex("Ay_minus") > -1,
           "Could not get index for Ay_minus.");
  ASSERTL1(this->GetFieldIndex("Az_minus") > -1,
           "Could not get index for Az_minus.");
  ASSERTL1(this->GetFieldIndex("Bx") > -1, "Could not get index for Bx.");
  ASSERTL1(this->GetFieldIndex("By") > -1, "Could not get index for By.");
  ASSERTL1(this->GetFieldIndex("Bz") > -1, "Could not get index for Bz.");
  ASSERTL1(this->GetFieldIndex("Ex") > -1, "Could not get index for Ex.");
  ASSERTL1(this->GetFieldIndex("Ey") > -1, "Could not get index for Ey.");
  ASSERTL1(this->GetFieldIndex("Ez") > -1, "Could not get index for Ez.");
}

int MaxwellWavePIC::GetFieldIndex(const std::string name) {
  ASSERTL1(this->field_to_index.count(name) > 0,
           "Could not map field name to index.");
  return this->field_to_index[name];
}

void MaxwellWavePIC::v_InitObject(bool DeclareFields) {
  EquationSystem::v_InitObject(true);
}

MaxwellWavePIC::~MaxwellWavePIC() {}

void MaxwellWavePIC::v_GenerateSummary(SolverUtils::SummaryList &s) {
  EquationSystem::SessionSummary(s);
}

Array<OneD, bool> MaxwellWavePIC::v_GetSystemSingularChecks() {
  auto singular_bools =
      Array<OneD, bool>(m_session->GetVariables().size(), false);
  singular_bools[this->GetFieldIndex("u")] = true;
  return singular_bools;
}

/**
 * Use time staggered formulation such that
 *       #  E.....E.....E
 *       #  B.....B.....B
 *       #  ϕ.....ϕ.....ϕ
 *       #  -..A..0..A..+..A
 *       #  ρ.....ρ.....ρ
 *       #  -..J..0..J..+..J
 *       #  x.....x.....x
 *       #  -..v..0..v..+..v
 *       #  past now future  : full timesteps
 *       #   -half  +half    : half timesteps
 *       #    time goes in this direction ->
 * Get E, B fields as stored
 *  - do boris push
 *  - advect half a timestep
 *  - deposit Jx, Jy, Jz
 *  - advect half a timestep
 *  - deposit rho
 * Solve Maxwell equations in wave form.
 * Replace f with phi, Ax, Ay, Az.
 * Replace s with rho, Jx, Jy, Jz.
 * ∇^2 f - ∂ₜ² f = - s
 * - (f⁺ - 2f⁰ + f⁻)/Δt^2 = - ∇^2 f - s
 * (f⁺ - 2f⁰ + f⁻) = Δt^2 (∇^2 f + s)
 * f⁺ = Δt^2 (∇^2 f + s) + 2f⁰ - f⁻
 * f⁺ = (2 + Δt^2 ∇²) f⁰ - f⁻ + Δt^2 s
 *
 * ∂ₜ² phi = ∇^2 phi + rho
 * ∂ₜ² Ax = ∇^2 Ax + Jx
 * ∂ₜ² Ay = ∇^2 Ay + Jy
 * ∂ₜ² Az = ∇^2 Az + Jz
 * 
 * At this point (ϕ, Ai) stores the (n+1)th timestep value and (ϕ⁻, Ai⁻) the nth
 * Now calculate the value of E and B at n+1/2
 * Eʰ = -∇(ϕ⁰ + ϕ⁺) / 2 - (A⁺ - A⁰)/dt
 * Bʰ = ∇x(A⁺ + A⁰)/2
 *
 * Don't forget to copy over phi_ -> phi, Jx_ -> Jx
 * - i.e. copy last timestep value of sources to be current value for the next
 * step
 */
void MaxwellWavePIC::v_DoSolve() {
  const int phi_index = this->GetFieldIndex("phi");
  const int phi_minus_index = this->GetFieldIndex("phi_minus");
  const int rho_index = this->GetFieldIndex("rho");
  const int Jx_index = this->GetFieldIndex("Jx");
  const int Jy_index = this->GetFieldIndex("Jy");
  const int Jz_index = this->GetFieldIndex("Jz");
  const int Jx_minus_index = this->GetFieldIndex("Jx_minus");
  const int Jy_minus_index = this->GetFieldIndex("Jy_minus");
  const int Jz_minus_index = this->GetFieldIndex("Jz_minus");
  const int Ax_index = this->GetFieldIndex("Ax");
  const int Ay_index = this->GetFieldIndex("Ay");
  const int Az_index = this->GetFieldIndex("Az");

  LorenzGuageSolve(phi_index, phi_minus_index, rho_index);
  LorenzGuageSolve(Jx_index, Jx_minus_index, Ax_index);
  LorenzGuageSolve(Jy_index, Jy_minus_index, Ay_index);
  LorenzGuageSolve(Jz_index, Jz_minus_index, Az_index);

  ElectricFieldSolve();
  MagneticFieldSolve();
}

void MaxwellWavePIC::ElectricFieldSolvePhi(const int E, const int phi,
                                            const int phi_minus,
                                            MultiRegions::Direction direction,
                                            int nPts) {
  auto Ephys = m_fields[E]->UpdatePhys();
  auto phiphys = m_fields[phi]->GetPhys();
  auto phi_1phys = m_fields[phi_minus]->GetPhys();
  Array<OneD, NekDouble> tempDeriv(nPts, 0.0);
  m_fields[phi]->PhysDeriv(direction, phiphys, tempDeriv); // tmp = ∇i ϕ⁺
  Vmath::Smul(nPts, -0.5, tempDeriv, 1, tempDeriv, 1); // tmp = -0.5 ∇i ϕ⁺
  // Vmath::Vadd(nPts, tempDeriv, 1, tempDeriv, 1, Ephys, 1); // Ex += tmp =
  // -0.5 ∇i ϕ⁺
  m_fields[phi_minus]->PhysDeriv(direction, phi_1phys,
                                 tempDeriv);           // tmp = ∇i ϕ⁰
  Vmath::Smul(nPts, -0.5, tempDeriv, 1, tempDeriv, 1); // tmp = -0.5 ∇i ϕ⁺
  Vmath::Vadd(nPts, tempDeriv, 1, tempDeriv, 1, Ephys,
              1); // Ei += tmp = -0.5 ∇i ϕ⁰
}

void MaxwellWavePIC::ElectricFieldSolveA(const int E, const int A,
                                          const int A_minus, int nPts,
                                          double one_dt) {
  auto Ephys = m_fields[E]->UpdatePhys();
  auto Aphys = m_fields[A]->GetPhys();
  auto A_1phys = m_fields[A_minus]->GetPhys();
  Vmath::Vsub(nPts, A_1phys, 1, Aphys, 1, Ephys, 1); // E = A0 - A1
  Vmath::Smul(nPts, one_dt, Ephys, 1, Ephys, 1);     // E *= 1 / dt
}

// Eʰ = -∇(ϕ⁰ + ϕ⁺) / 2 - (A⁺ - A⁰)/dt
void MaxwellWavePIC::ElectricFieldSolve() {
  const int phi = this->GetFieldIndex("phi"); // currently holds ϕ⁺
  const int phi_minus = this->GetFieldIndex("phi_minus"); // currently holds ϕ⁰
  const int Ax = this->GetFieldIndex("Ax");             // currently holds Ax⁺
  const int Ax_minus = this->GetFieldIndex("Ax_minus"); // currently holds Ax⁰
  const int Ay = this->GetFieldIndex("Ay");             // currently holds Ay⁺
  const int Ay_minus = this->GetFieldIndex("Ay_minus"); // currently holds Ay⁰
  const int Az = this->GetFieldIndex("Az");             // currently holds Az⁺
  const int Az_minus = this->GetFieldIndex("Az_minus"); // currently holds Az⁰
  const int Ex = this->GetFieldIndex("Ex");
  const int Ey = this->GetFieldIndex("Ey");
  const int Ez = this->GetFieldIndex("Ez");

  int nPts = GetNpoints();
  double one_dt = 1.0 / m_timestep;

  ElectricFieldSolveA(Ex, Ax, Ax_minus, nPts, one_dt);
  ElectricFieldSolveA(Ey, Ay, Ay_minus, nPts, one_dt);
  ElectricFieldSolveA(Ez, Az, Az_minus, nPts, one_dt);
  ElectricFieldSolvePhi(Ex, phi, phi_minus, MultiRegions::eX, nPts);
  ElectricFieldSolvePhi(Ey, phi, phi_minus, MultiRegions::eY, nPts);
  // no z component because it's 2D
}

// Bʰ = ∇x(A⁺ + A⁰)/2
void MaxwellWavePIC::MagneticFieldSolveCurl(const int Bx, const int By,
                                             const int Bz, int nPts) {
  Array<OneD, NekDouble> dAzdy(nPts, 0.0);
  m_fields[Bz]->PhysDeriv(MultiRegions::eY, m_fields[Bz]->GetPhys(), dAzdy);
  Vmath::Vadd(nPts, dAzdy, 1, m_fields[Bx]->GetPhys(), 1,
              m_fields[Bx]->UpdatePhys(), 1);
  // reuse array but change name to make it more readable
  auto &dAzdx = dAzdy;
  m_fields[Bz]->PhysDeriv(MultiRegions::eX, m_fields[Bz]->GetPhys(), dAzdx);
  Vmath::Vsub(nPts, dAzdy, 1, m_fields[By]->GetPhys(), 1,
              m_fields[By]->UpdatePhys(), 1);

  auto &dAydx = dAzdx;
  m_fields[By]->PhysDeriv(MultiRegions::eX, m_fields[By]->GetPhys(), dAydx);
  Vmath::Vadd(nPts, dAydx, 1, m_fields[Bz]->GetPhys(), 1,
              m_fields[Bz]->UpdatePhys(), 1);
  auto &dAxdy = dAydx;
  m_fields[Bx]->PhysDeriv(MultiRegions::eY, m_fields[Bx]->GetPhys(), dAxdy);
  Vmath::Vsub(nPts, dAxdy, 1, m_fields[Bz]->GetPhys(), 1,
              m_fields[Bz]->UpdatePhys(), 1);
}

void MaxwellWavePIC::MagneticFieldSolve() {
  const int Ax = this->GetFieldIndex("Ax");             // currently holds Ax⁺
  const int Ax_minus = this->GetFieldIndex("Ax_minus"); // currently holds Ax⁰
  const int Ay = this->GetFieldIndex("Ay");             // currently holds Ay⁺
  const int Ay_minus = this->GetFieldIndex("Ay_minus"); // currently holds Ay⁰
  const int Az = this->GetFieldIndex("Az");             // currently holds Az⁺
  const int Az_minus = this->GetFieldIndex("Az_minus"); // currently holds Az⁰
  const int Bx = this->GetFieldIndex("Bx");
  const int By = this->GetFieldIndex("By");
  const int Bz = this->GetFieldIndex("Bz");
  int nPts = GetNpoints();

  Vmath::Zero(nPts, m_fields[Bx]->UpdatePhys(), 1);
  Vmath::Zero(nPts, m_fields[By]->UpdatePhys(), 1);
  Vmath::Zero(nPts, m_fields[Bz]->UpdatePhys(), 1);
  MagneticFieldSolveCurl(Ax, Ay, Az, nPts);
  MagneticFieldSolveCurl(Ax_minus, Ay_minus, Az_minus, nPts);
  Vmath::Smul(nPts, 0.5, m_fields[Bx]->GetPhys(), 1, m_fields[Bx]->UpdatePhys(),
              1);
  Vmath::Smul(nPts, 0.5, m_fields[By]->GetPhys(), 1, m_fields[By]->UpdatePhys(),
              1);
  Vmath::Smul(nPts, 0.5, m_fields[Bz]->GetPhys(), 1, m_fields[Bz]->UpdatePhys(),
              1);
}

void MaxwellWavePIC::LorenzGuageSolve(const int field_t_index,
                                       const int field_t_minus1_index,
                                       const int source_index) {
  // copy across into shorter variable names to make sure code fits
  // on one line, more readable that way.
  const int f0 = field_t_index;
  const int f_1 = field_t_minus1_index;
  const int s = source_index;
  const int nPts = GetNpoints();
  const double dt2 = std::pow(m_timestep, 2);

  auto f0phys = m_fields[f0]->UpdatePhys();
  auto f_1phys = m_fields[f_1]->UpdatePhys();
  auto sphys = m_fields[s]->UpdatePhys();

  // f⁺ = (2 + Δt^2 ∇²) f⁰ - f⁻ + Δt^2 s
  Array<OneD, NekDouble> tempDerivX(nPts, 0.0);
  Array<OneD, NekDouble> tempDerivY(nPts, 0.0);
  Array<OneD, NekDouble> tempLaplacian(nPts, 0.0);
  m_fields[f0]->PhysDeriv(MultiRegions::eX, f0phys, tempDerivX);
  m_fields[f0]->PhysDeriv(MultiRegions::eX, tempDerivX, tempDerivX);
  m_fields[f0]->PhysDeriv(MultiRegions::eY, f0phys, tempDerivY);
  m_fields[f0]->PhysDeriv(MultiRegions::eY, tempDerivY, tempDerivY);
  Vmath::Vadd(nPts, tempDerivX, 1, tempDerivY, 1, tempLaplacian, 1); // ∇² f0
  Vmath::Smul(nPts, dt2, tempLaplacian, 1, tempLaplacian, 1); // Δt^2 ∇² f0
  Vmath::Smul(nPts, dt2, sphys, 1, sphys,
              1); // s = Δt^2 s // sphys now hold Δt^2 s
  Vmath::Vsub(nPts, sphys, 1, f_1phys, 1, sphys,
              1); // s -= f_1 // sphys now holds Δt^2 s - f_1
  Vmath::Vcopy(nPts, f0phys, 1, f_1phys, 1);    // f_1 -> f0 // f_1 now holds f0
  Vmath::Smul(nPts, 2.0, f0phys, 1, f0phys, 1); // f0 = 2 f0 // f0 now holds 2f0
  Vmath::Vadd(nPts, f0phys, 1, tempLaplacian, 1, f0phys,
              1); // f0 now holds 2f0 + Δt^2 ∇² f0
  Vmath::Vadd(nPts, f0phys, 1, sphys, 1, f0phys,
              1); // f0 now holds 2f0 + Δt^2 ∇² f0 + Δt^2 s - f_1
  Vmath::Zero(m_fields[s]->GetNcoeffs(), m_fields[s]->UpdateCoeffs(),
              1); // s now holds 0
}

} // namespace Nektar
