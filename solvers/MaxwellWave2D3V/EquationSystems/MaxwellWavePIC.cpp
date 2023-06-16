#include "MaxwellWavePIC.h"

namespace Nektar {
std::string MaxwellWavePIC::className1 =
    GetEquationSystemFactory().RegisterCreatorFunction("MaxwellWavePIC",
                                                       MaxwellWavePIC::create);

MaxwellWavePIC::MaxwellWavePIC(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph), m_factors(), m_DtMultiplier(1.0) {

  ASSERTL1(m_timestep > 0,
           "TimeStep must be set and be > 0 in the xml config file.");

  double lengthScale;
  pSession->LoadParameter("length_scale", lengthScale);

  m_unitConverter = std::make_shared<UnitConverter>(lengthScale);

  m_factors[StdRegions::eFactorLambda] = 0.0;
  m_factors[StdRegions::eFactorTau] = 1.0;
  auto variables = pSession->GetVariables();
  int index = 0;
  for (auto vx : variables) {
    this->field_to_index[vx] = index;
    index++;
  }

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

  pSession->LoadParameter("B0x", m_B0x);
  pSession->LoadParameter("B0y", m_B0y);
  pSession->LoadParameter("B0z", m_B0z);
  m_B0x = m_unitConverter->si_magneticfield_to_sim(m_B0x);
  m_B0y = m_unitConverter->si_magneticfield_to_sim(m_B0y);
  m_B0z = m_unitConverter->si_magneticfield_to_sim(m_B0z);
}

int MaxwellWavePIC::GetFieldIndex(const std::string name) {
  ASSERTL1(this->field_to_index.count(name) > 0,
           "Could not map field name to index.");
  return this->field_to_index[name];
}

double MaxwellWavePIC::timeStep() { return m_DtMultiplier * m_timestep; }

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
  singular_bools[this->GetFieldIndex("phi")] = true;
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
  const int Ax_index = this->GetFieldIndex("Ax");
  const int Ay_index = this->GetFieldIndex("Ay");
  const int Az_index = this->GetFieldIndex("Az");
  const int Ax_minus_index = this->GetFieldIndex("Ax_minus");
  const int Ay_minus_index = this->GetFieldIndex("Ay_minus");
  const int Az_minus_index = this->GetFieldIndex("Az_minus");
  const int Jx_index = this->GetFieldIndex("Jx");
  const int Jy_index = this->GetFieldIndex("Jy");
  const int Jz_index = this->GetFieldIndex("Jz");

  LorenzGuageSolve(phi_index, phi_minus_index, rho_index);
  LorenzGuageSolve(Ax_index, Ax_minus_index, Jx_index);
  LorenzGuageSolve(Ay_index, Ay_minus_index, Jy_index);
  LorenzGuageSolve(Az_index, Az_minus_index, Jz_index);

  // std::cout << "Jphys x" << std::endl;
  // for (const auto i : m_fields[Jx_index]->GetPhys()) { std::cout << i <<
  // std::endl; } std::cout << "Jphys y" << std::endl; for (const auto i :
  // m_fields[Jy_index]->GetPhys()) { std::cout << i << std::endl; } std::cout
  // << "Jphys z" << std::endl; for (const auto i :
  // m_fields[Jz_index]->GetPhys()) { std::cout << i << std::endl; }

  ElectricFieldSolve();
  MagneticFieldSolve();
}

void MaxwellWavePIC::setDtMultiplier(const double dtMultiplier) {
  m_DtMultiplier = dtMultiplier;
}

void MaxwellWavePIC::setTheta(const double theta) {
  ASSERTL1(0 <= theta,
           "Theta (0 = explicit, 1=implicit) must not be negative.");
  ASSERTL1(theta <= 1,
           "Theta (0 = explicit, 1=implicit) must not be greater than 1.");
  m_theta = theta;
}

void MaxwellWavePIC::ElectricFieldSolvePhi(const int E, const int phi,
                                           const int phi_minus,
                                           MultiRegions::Direction direction,
                                           const int nPts) {
  auto Ephys = m_fields[E]->UpdatePhys();
  auto phiphys = m_fields[phi]->GetPhys();
  auto phi_1phys = m_fields[phi_minus]->GetPhys();
  Array<OneD, NekDouble> tempDeriv(nPts, 0.0);
  m_fields[phi]->PhysDeriv(direction, phiphys, tempDeriv); // tmp = ∇i ϕ⁺
  Vmath::Vsub(nPts, Ephys, 1, tempDeriv, 1, Ephys, 1);     // Ex += - ∇i ϕ⁺
  //  Vmath::Smul(nPts, -0.5, tempDeriv, 1, tempDeriv, 1); // tmp = -0.5 ∇i ϕ⁺
  //  Vmath::Vadd(nPts, tempDeriv, 1, Ephys, 1, Ephys, 1); // Ex += -0.5 ∇i ϕ⁺
  //  m_fields[phi_minus]->PhysDeriv(direction, phi_1phys,
  //                                 tempDeriv);           // tmp = ∇i ϕ⁰
  //  Vmath::Smul(nPts, -0.5, tempDeriv, 1, tempDeriv, 1); // tmp = -0.5 ∇i ϕ⁰
  //  Vmath::Vadd(nPts, tempDeriv, 1, Ephys, 1, Ephys, 1); // Ei += -0.5 ∇i ϕ⁰
}

void MaxwellWavePIC::ElectricFieldSolveA(const int E_index, const int A1_index,
                                         const int A0_index, const int nPts) {
  double dt = timeStep();
  auto Ephys = m_fields[E_index]->UpdatePhys();
  auto A1phys = m_fields[A1_index]->GetPhys();
  auto A0phys = m_fields[A0_index]->GetPhys();
  Vmath::Vsub(nPts, A0phys, 1, A1phys, 1, Ephys, 1); // E = A0 - A1
  Vmath::Smul(nPts, 1.0 / dt, Ephys, 1, Ephys, 1);   // E *= 1 / dt
}

// Eʰ = -∇(ϕ⁰ + ϕ⁺) / 2 - (A⁺ - A⁰)/dt
void MaxwellWavePIC::ElectricFieldSolve() {
  const int Ax = this->GetFieldIndex("Ax");             // currently holds Ax⁺
  const int Ay = this->GetFieldIndex("Ay");             // currently holds Ay⁺
  const int Az = this->GetFieldIndex("Az");             // currently holds Az⁺
  const int phi = this->GetFieldIndex("phi");           // currently holds ϕ⁺
  const int Ax_minus = this->GetFieldIndex("Ax_minus"); // currently holds Ax⁰
  const int Ay_minus = this->GetFieldIndex("Ay_minus"); // currently holds Ay⁰
  const int Az_minus = this->GetFieldIndex("Az_minus"); // currently holds Az⁰
  const int phi_minus = this->GetFieldIndex("phi_minus"); // currently holds ϕ⁰
  const int Ex = this->GetFieldIndex("Ex");
  const int Ey = this->GetFieldIndex("Ey");
  const int Ez = this->GetFieldIndex("Ez");

  const int nPts = GetNpoints();

  ElectricFieldSolveA(Ex, Ax, Ax_minus, nPts);
  ElectricFieldSolveA(Ey, Ay, Ay_minus, nPts);
  ElectricFieldSolveA(Ez, Az, Az_minus, nPts);
  ElectricFieldSolvePhi(Ex, phi, phi_minus, MultiRegions::eX, nPts);
  ElectricFieldSolvePhi(Ey, phi, phi_minus, MultiRegions::eY, nPts);
  // no ElecticFieldSolvePhi for Ez because it's 2D; hence no derivative in z

  m_fields[Ex]->FwdTrans(m_fields[Ex]->GetPhys(), m_fields[Ex]->UpdateCoeffs());
  m_fields[Ey]->FwdTrans(m_fields[Ey]->GetPhys(), m_fields[Ey]->UpdateCoeffs());
  m_fields[Ez]->FwdTrans(m_fields[Ez]->GetPhys(), m_fields[Ez]->UpdateCoeffs());

  //  std::cout << "Ephys x" << std::endl;
  //  for (const auto i : m_fields[Ex]->GetPhys()) { std::cout << i <<
  //  std::endl; } std::cout << "Ephys y" << std::endl; for (const auto i :
  //  m_fields[Ey]->GetPhys()) { std::cout << i << std::endl; } std::cout <<
  //  "Ephys z" << std::endl; for (const auto i : m_fields[Ez]->GetPhys()) {
  //  std::cout << i << std::endl; }
}

// Bʰ = ∇x(A⁺ + A⁰)/2
void MaxwellWavePIC::MagneticFieldSolveCurl(const int Ax, const int Ay,
                                            const int Az, const int nPts) {
  const int Bx = this->GetFieldIndex("Bx");
  const int By = this->GetFieldIndex("By");
  const int Bz = this->GetFieldIndex("Bz");
  Array<OneD, NekDouble> dAzdy(nPts, 0.0);
  m_fields[Az]->PhysDeriv(MultiRegions::eY, m_fields[Az]->GetPhys(), dAzdy);
  Vmath::Vadd(nPts, dAzdy, 1, m_fields[Bx]->GetPhys(), 1,
              m_fields[Bx]->UpdatePhys(), 1); // Bx = dAz/dy + Bx
  // reuse array but change name to make it more readable
  auto &dAzdx = dAzdy;
  m_fields[Az]->PhysDeriv(MultiRegions::eX, m_fields[Az]->GetPhys(), dAzdx);
  Vmath::Vsub(nPts, m_fields[By]->GetPhys(), 1, dAzdy, 1,
              m_fields[By]->UpdatePhys(), 1); // By = By - dAz/dx

  auto &dAydx = dAzdx;
  m_fields[Ay]->PhysDeriv(MultiRegions::eX, m_fields[Ay]->GetPhys(), dAydx);
  Vmath::Vadd(nPts, dAydx, 1, m_fields[Bz]->GetPhys(), 1,
              m_fields[Bz]->UpdatePhys(), 1); // Bz = dAz/dx + Bz
  auto &dAxdy = dAydx;
  m_fields[Ax]->PhysDeriv(MultiRegions::eY, m_fields[Ax]->GetPhys(), dAxdy);
  Vmath::Vsub(nPts, m_fields[Bz]->GetPhys(), 1, dAxdy, 1,
              m_fields[Bz]->UpdatePhys(), 1); // Bz = Bz - dAx/dy
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
  const int nPts = GetNpoints();

  Vmath::Zero(nPts, m_fields[Bx]->UpdatePhys(), 1);
  Vmath::Zero(nPts, m_fields[By]->UpdatePhys(), 1);
  Vmath::Zero(nPts, m_fields[Bz]->UpdatePhys(), 1);
  // Bx = By = Bz = 0
  MagneticFieldSolveCurl(Ax, Ay, Az, nPts);
  // B fields have values of ∇xA⁺
  MagneticFieldSolveCurl(Ax_minus, Ay_minus, Az_minus, nPts);
  // B fields have values of ∇x(A⁺ + A⁰)
  Vmath::Smul(nPts, 0.5, m_fields[Bx]->GetPhys(), 1, m_fields[Bx]->UpdatePhys(),
              1);
  Vmath::Smul(nPts, 0.5, m_fields[By]->GetPhys(), 1, m_fields[By]->UpdatePhys(),
              1);
  Vmath::Smul(nPts, 0.5, m_fields[Bz]->GetPhys(), 1, m_fields[Bz]->UpdatePhys(),
              1);
  // B fields have values of ∇x(A⁺ + A⁰)/2
  // Add the static equilibrium magnetic field values on
  Vmath::Sadd(nPts, m_B0x, m_fields[Bx]->GetPhys(), 1,
              m_fields[Bx]->UpdatePhys(), 1);
  Vmath::Sadd(nPts, m_B0y, m_fields[By]->GetPhys(), 1,
              m_fields[By]->UpdatePhys(), 1);
  Vmath::Sadd(nPts, m_B0z, m_fields[Bz]->GetPhys(), 1,
              m_fields[Bz]->UpdatePhys(), 1);

  m_fields[Bx]->FwdTrans(m_fields[Bx]->GetPhys(), m_fields[Bx]->UpdateCoeffs());
  m_fields[By]->FwdTrans(m_fields[By]->GetPhys(), m_fields[By]->UpdateCoeffs());
  m_fields[Bz]->FwdTrans(m_fields[Bz]->GetPhys(), m_fields[Bz]->UpdateCoeffs());

  //  std::cout << "Bphys x" << std::endl;
  //  for (const auto i : m_fields[Bx]->GetPhys()) { std::cout << i <<
  //  std::endl; } std::cout << "Bphys y" << std::endl; for (const auto i :
  //  m_fields[By]->GetPhys()) { std::cout << i << std::endl; } std::cout <<
  //  "Bphys z" << std::endl; for (const auto i : m_fields[Bz]->GetPhys()) {
  //  std::cout << i << std::endl; }
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
  const double dt2 = std::pow(timeStep(), 2);

  auto f0phys = m_fields[f0]->UpdatePhys();
  auto f_1phys = m_fields[f_1]->UpdatePhys();
  auto sphys = m_fields[s]->GetPhys();

  Array<OneD, NekDouble> tempDerivX(nPts, 0.0);
  Array<OneD, NekDouble> tempDerivY(nPts, 0.0);
  Array<OneD, NekDouble> rhs(nPts, 0.0);
  m_fields[f0]->PhysDeriv(MultiRegions::eX, m_fields[f0]->GetPhys(),
                          tempDerivX);
  m_fields[f0]->PhysDeriv(MultiRegions::eX, tempDerivX, tempDerivX);
  m_fields[f0]->PhysDeriv(MultiRegions::eY, m_fields[f0]->GetPhys(),
                          tempDerivY);
  m_fields[f0]->PhysDeriv(MultiRegions::eY, tempDerivY, tempDerivY);
  Vmath::Vadd(nPts, tempDerivX, 1, tempDerivY, 1, rhs, 1); // rhs = ∇² f0

  if (m_theta == 0.0) {
    //  std::cout << "dt2 = " << dt2 << std::endl;
    //  std::cout << "sphys" << std::endl;
    //  for (auto ij : sphys) {
    //    std::cout << ij << std::endl;
    //  }
    //
    //  std::cout << "f_1phys" << std::endl;
    //  for (auto ij : f_1phys) {
    //    std::cout << ij << std::endl;
    //  }
    //
    //  std::cout << "f0phys" << std::endl;
    //  for (auto ij : f0phys) {
    //    std::cout << ij << std::endl;
    //  }

    // f⁺ = (2 + Δt^2 ∇²) f⁰ - f⁻ + Δt^2 s
    Vmath::Smul(nPts, dt2, rhs, 1, rhs,
                1); // rhs = Δt^2 ∇² f0
    Array<OneD, NekDouble> work(nPts, 0.0);
    Vmath::Smul(nPts, dt2, sphys, 1, work,
                1); // work = Δt^2 s // work now holds Δt^2 s
    Vmath::Vsub(nPts, work, 1, f_1phys, 1, work,
                1); // s -= f_1 // work now holds Δt^2 s - f_1
    Vmath::Vcopy(nPts, f0phys, 1, f_1phys,
                 1); // f_1 -> f0 // f_1 now holds f0 (phys values)
    Vmath::Smul(nPts, 2.0, f0phys, 1, f0phys,
                1); // f0 = 2 f0 // f0 now holds 2f0
    Vmath::Vadd(nPts, f0phys, 1, rhs, 1, f0phys,
                1); // f0 now holds 2f0 + Δt^2 ∇² f0
    Vmath::Vadd(nPts, f0phys, 1, work, 1, f0phys,
                1); // f0 now holds 2f0 + Δt^2 ∇² f0 + Δt^2 s - f_1

    //  std::cout << "f1phys" << std::endl;
    //  for (auto ij : f0phys) {
    //    std::cout << ij << std::endl;
    //  }

    // Copy f_1 coefficients to f0 (no need to solve again!) ((N.B. phys values
    // copied across above)) N.B. phys values were copied above
    Vmath::Vcopy(nPts, m_fields[f0]->GetCoeffs(), 1,
                 m_fields[f_1]->UpdateCoeffs(), 1);
    //double maxabs = 0.0;
    //for (auto i : f0phys) {
    //  maxabs = std::max(maxabs, std::abs(i));
    //}
    //double maxabs2 = 0.0;
    //for (auto i : m_fields[f0]->UpdateCoeffs()) {
    //  maxabs2 = std::max(maxabs2, std::abs(i));
    //}
    //std::cout << "before maxabs f0phys = " << maxabs << std::endl;
    //std::cout << "before maxabs updatecoeffs = " << maxabs2 << std::endl;
    m_fields[f0]->FwdTrans(f0phys, m_fields[f0]->UpdateCoeffs());
    //std::cout << "after maxabs f0phys = " << maxabs << std::endl;
    //std::cout << "after maxabs updatecoeffs = " << maxabs2 << std::endl;

  } else {

    double lambda = 2.0 / dt2 / m_theta;
    Vmath::Smul(nPts, -2 * (1 - m_theta) / m_theta, rhs, 1, rhs, 1);
    // Svtvp (n, a, x, _, y, _, z, _) -> z = a * x + y
    Vmath::Svtvp(nPts, -2 * lambda, f0phys, 1, rhs, 1, rhs,
                 1); // rhs now holds the f0 rhs values

    Array<OneD, NekDouble> rhs_a(nPts, 0.0);
    m_fields[f0]->PhysDeriv(MultiRegions::eX, m_fields[f_1]->GetPhys(),
                            tempDerivX);
    m_fields[f0]->PhysDeriv(MultiRegions::eX, tempDerivX, tempDerivX);
    m_fields[f0]->PhysDeriv(MultiRegions::eY, m_fields[f_1]->GetPhys(),
                            tempDerivY);
    m_fields[f0]->PhysDeriv(MultiRegions::eY, tempDerivY, tempDerivY);
    Vmath::Vadd(nPts, tempDerivX, 1, tempDerivY, 1, rhs_a, 1); // rhs_a = ∇² f_1

    // Svtvp (n, a, x, _, y, _, z, _) -> z = a * x + y
    Vmath::Svtvp(nPts, -lambda, f_1phys, 1, rhs_a, 1, rhs_a, 1);
    Vmath::Vsub(nPts, rhs, 1, rhs_a, 1, rhs,
                1); // rhs now holds the f0 and f_1 rhs values

    Vmath::Smul(nPts, -2.0 / m_theta, sphys, 1, rhs_a,
                1); // rhs_a now holds the source term
    Vmath::Vadd(nPts, rhs_a, 1, rhs, 1, rhs,
                1); // rhs now has the f0 and source term

    StdRegions::ConstFactorMap factors;
    factors[StdRegions::eFactorLambda] = lambda; // TODO ?
    factors[StdRegions::eFactorTau] = 0.0;       // TODO ?

    // copy f_1 coefficients to f0 (no need to solve again!) ((N.B. phys values
    // copied across above))
    Vmath::Vcopy(nPts, m_fields[f0]->GetPhys(), 1, m_fields[f_1]->UpdatePhys(),
                 1);
    Vmath::Vcopy(nPts, m_fields[f0]->GetCoeffs(), 1,
                 m_fields[f_1]->UpdateCoeffs(), 1);

    // TODO: are the Phys / Coeffs right here?
    m_fields[f0]->HelmSolve(rhs, m_fields[f0]->UpdateCoeffs(), factors);
    // TODO: may need correction based on use of Phys / Coeffs from HelmSolve
    m_fields[f0]->BwdTrans(m_fields[f0]->GetCoeffs(),
                           m_fields[f0]->UpdatePhys());
  }
}

} // namespace Nektar
