#include "MaxwellWaveSystem.h"

namespace Nektar {
std::string MaxwellWaveSystem::className =
    GetEquationSystemFactory().RegisterCreatorFunction("MaxwellWaveSystem",
                                                       MaxwellWaveSystem::create);

MaxwellWaveSystem::MaxwellWaveSystem(
    const LibUtilities::SessionReaderSharedPtr &pSession,
    const SpatialDomains::MeshGraphSharedPtr &pGraph)
    : EquationSystem(pSession, pGraph), m_factors(), m_DtMultiplier(1.0) {

  double lengthScale;
  pSession->LoadParameter("length_scale", lengthScale);

  m_factors[StdRegions::eFactorTau] = 1.0;

  m_unitConverter = std::make_shared<UnitConverter>(lengthScale);

  // m_factors[StdRegions::eFactorLambda] = 0.0;
  m_factors[StdRegions::eFactorTau] = 1.0;
  auto variables = pSession->GetVariables();
  int index = 0;
  for (auto vx : variables) {
    this->field_to_index[vx] = index;
    index++;
  }

  ASSERTL1(this->GetFieldIndex("rho") > -1, "Could not get index for rho.");
  ASSERTL1(this->GetFieldIndex("rho_minus") > -1, "Could not get index for rho.");
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

  m_volume = -1.0;

  this->m_session->LoadParameter("Theta", this->m_theta, 0.0);
  this->m_session->LoadParameter("TimeStep", this->m_timestep);

  ASSERTL1(m_theta >= 0,
           "Theta must be set and be >= 0 in the xml config file.");
  ASSERTL1(m_timestep > 0,
           "TimeStep must be set and be > 0 in the xml config file.");

  m_perform_charge_conservation = true;
}

void MaxwellWaveSystem::SetVolume(const double volume) {
  m_volume = volume;
}

void MaxwellWaveSystem::ChargeConservationSwitch(const bool onoff) {
  m_perform_charge_conservation = onoff;
}

int MaxwellWaveSystem::GetFieldIndex(const std::string name) {
  ASSERTL1(this->field_to_index.count(name) > 0,
           "Could not map field name to index.");
  return this->field_to_index[name];
}

double MaxwellWaveSystem::timeStep() { return m_DtMultiplier * m_timestep; }

void MaxwellWaveSystem::v_InitObject(bool DeclareFields) {
  EquationSystem::v_InitObject(DeclareFields);
  for (auto f : m_fields) {
    ASSERTL1(f->GetNpoints() > 0, "GetNpoints must return > 0");
  }

  // Read ICs from the file
  const int domain = 0; // if this is different to the DOMAIN in the mesh it segfaults.
  this->SetInitialConditions(0.0, true, domain);

  for (auto f : m_fields) {
    ASSERTL1(f->GetNpoints() > 0, "GetNpoints must return > 0");
    ASSERTL1(f->GetNcoeffs() > 0, "GetNcoeffs must return > 0");
  }

  // Set up diffusion object
  std::string diff_type;
  m_session->LoadSolverInfo("DiffusionType", diff_type, "LDG");
  m_diffusion = GetDiffusionFactory().CreateInstance(diff_type, diff_type);
  m_diffusion->SetFluxVector(&MaxwellWaveSystem::GetDiffusionFluxVector, this);
  m_diffusion->InitObject(m_session, m_fields);
}

MaxwellWaveSystem::~MaxwellWaveSystem() {}

void MaxwellWaveSystem::v_GenerateSummary(SolverUtils::SummaryList &s) {
  EquationSystem::SessionSummary(s);
}

Array<OneD, bool> MaxwellWaveSystem::v_GetSystemSingularChecks() {
  auto singular_bools =
      Array<OneD, bool>(m_session->GetVariables().size(), false);
  //singular_bools[this->GetFieldIndex("phi")] = true;
  //singular_bools[this->GetFieldIndex("Ax")] = true;
  //singular_bools[this->GetFieldIndex("Ay")] = true;
  //singular_bools[this->GetFieldIndex("Az")] = true;
  // The "minus" fields, which hold the previous timestep's solution,
  // are also singular fields and are marked so for completeness (and out of
  // paranoia) even though they should never be used in a solve.
  //singular_bools[this->GetFieldIndex("phi_minus")] = true;
  //singular_bools[this->GetFieldIndex("Ax_minus")] = true;
  //singular_bools[this->GetFieldIndex("Ay_minus")] = true;
  //singular_bools[this->GetFieldIndex("Az_minus")] = true;
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
void MaxwellWaveSystem::v_DoSolve() {
  const int phi_index = this->GetFieldIndex("phi");
  const int phi_minus_index = this->GetFieldIndex("phi_minus");
  const int rho_index = this->GetFieldIndex("rho");
  const int rho_minus_index = this->GetFieldIndex("rho_minus");
  const int Ax_index = this->GetFieldIndex("Ax");
  const int Ay_index = this->GetFieldIndex("Ay");
  const int Az_index = this->GetFieldIndex("Az");
  const int Ax_minus_index = this->GetFieldIndex("Ax_minus");
  const int Ay_minus_index = this->GetFieldIndex("Ay_minus");
  const int Az_minus_index = this->GetFieldIndex("Az_minus");
  const int Jx_index = this->GetFieldIndex("Jx");
  const int Jy_index = this->GetFieldIndex("Jy");
  const int Jz_index = this->GetFieldIndex("Jz");

  SubtractMean(Jx_index);
  SubtractMean(Jy_index);
  SubtractMean(Jz_index);
  if (this->m_perform_charge_conservation) {
    ChargeConservation(rho_index, rho_minus_index, Jx_index, Jy_index);
  }
  SubtractMean(rho_index);
  LorenzGaugeSolve(phi_index, phi_minus_index, rho_index);
  LorenzGaugeSolve(Ax_index, Ax_minus_index, Jx_index);
  LorenzGaugeSolve(Ay_index, Ay_minus_index, Jy_index);
  LorenzGaugeSolve(Az_index, Az_minus_index, Jz_index);

  // TODO: figure out how to get inter-cell interactions etc and make
  // the grid "physical"

  ElectricFieldSolve();
  MagneticFieldSolve();
}

void MaxwellWaveSystem::SubtractMean(const int field_index) {
  auto field = this->m_fields[field_index];
  // Nektar reduces the integral across all ranks
  double integral = field->Integral();
  ASSERTL1(m_volume > 0, "Volume has not be set correctly. It must be > 0");
  const double mean = integral / m_volume;
  const int nPts = GetNpoints();
  //std::cout << "integral, mean, volume = " << integral << ", " << mean << ", " << m_volume << std::endl;
  Vmath::Sadd(nPts, -mean, field->GetPhys(), 1, field->UpdatePhys(), 1);
  field->FwdTrans(field->GetPhys(), field->UpdateCoeffs());
}

void MaxwellWaveSystem::setDtMultiplier(const double dtMultiplier) {
  m_DtMultiplier = dtMultiplier;
}

void MaxwellWaveSystem::setTheta(const double theta) {
  ASSERTL1(0 <= theta,
           "Theta (0 = explicit, 1=implicit) must not be negative.");
  ASSERTL1(theta <= 1,
           "Theta (0 = explicit, 1=implicit) must not be greater than 1.");
  m_theta = theta;
}

/**
 * @brief Return the flux vector for the unsteady diffusion problem.
 */
void MaxwellWaveSystem::GetDiffusionFluxVector(
    const Array<OneD, Array<OneD, NekDouble>> &in_arr,
    const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &q_field,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscous_tensor) {
  boost::ignore_unused(in_arr);

  unsigned int nDim = q_field.size();
  unsigned int nConvectiveFields = q_field[0].size();
  unsigned int nPts = q_field[0][0].size();

  // Hard-code diff coeffs
  NekDouble d[2] = {1.0, 1.0};

  for (unsigned int j = 0; j < nDim; ++j) {
    for (unsigned int i = 0; i < nConvectiveFields; ++i) {
      Vmath::Smul(nPts, d[j], q_field[j][i], 1, viscous_tensor[j][i], 1);
    }
  }
}


void MaxwellWaveSystem::ElectricFieldSolvePhi(const int E,
                                              const int phi,
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

void MaxwellWaveSystem::ElectricFieldSolveA(const int E_index, const int A1_index,
                                         const int A0_index, const int nPts) {
  double dt = timeStep();
  auto Ephys = m_fields[E_index]->UpdatePhys();
  auto A1phys = m_fields[A1_index]->GetPhys();
  auto A0phys = m_fields[A0_index]->GetPhys();
  Vmath::Vsub(nPts, A0phys, 1, A1phys, 1, Ephys, 1); // E = A0 - A1
  Vmath::Smul(nPts, 1.0 / dt, Ephys, 1, Ephys, 1);   // E *= 1 / dt
}

// Eʰ = -∇(ϕ⁰ + ϕ⁺) / 2 - (A⁺ - A⁰)/dt
void MaxwellWaveSystem::ElectricFieldSolve() {
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

  // // set to zero for debugging
  // Vmath::Zero(nPts, m_fields[Ex]->UpdatePhys(), 1);
  // Vmath::Zero(nPts, m_fields[Ey]->UpdatePhys(), 1);
  // Vmath::Zero(nPts, m_fields[Ey]->UpdatePhys(), 1);
  // Vmath::Zero(nPts, m_fields[Ex]->UpdateCoeffs(), 1);
  // Vmath::Zero(nPts, m_fields[Ey]->UpdateCoeffs(), 1);
  // Vmath::Zero(nPts, m_fields[Ey]->UpdateCoeffs(), 1);
}

// Bʰ = ∇x(A⁺ + A⁰)/2
void MaxwellWaveSystem::MagneticFieldSolveCurl(const int Ax, const int Ay,
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

void MaxwellWaveSystem::MagneticFieldSolve() {
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
}

void MaxwellWaveSystem::ChargeConservation(const int rho_index,
                                           const int rho_minus_index,
                                           const int jx_index,
                                           const int jy_index) {
  const int nPts = GetNpoints();
  const double dt = timeStep();
  auto rho = m_fields[rho_index]->UpdatePhys();
  auto rho_1 = m_fields[rho_minus_index]->UpdatePhys();
  auto jx = m_fields[jx_index]->GetPhys();
  auto jy = m_fields[jy_index]->GetPhys();
  // solve rho = rho_1 - dt *(d_dx jx + d_dy Jy);
  Vmath::Vcopy(nPts, rho_1, 1, rho, 1); // rho = rho_1

  Array<OneD, NekDouble> temp(nPts, 0.0);
  // temp = d_dx jx
  m_fields[jx_index]->PhysDeriv(MultiRegions::eX, jx, temp);

  // Svtvp (n, a, x, _, y, _, z, _) -> z = a * x + y
  // rho = rho_1 - dt d_dx jx
  Vmath::Svtvp(nPts, -dt, temp, 1, rho, 1, rho, 1);
  // temp = d_dy jy
  m_fields[jy_index]->PhysDeriv(MultiRegions::eY, jy, temp);
  // rho = rho_1 - dt d_dx jx - dt d_dy jy
  Vmath::Svtvp(nPts, -dt, temp, 1, rho, 1, rho, 1);

  // copy for next loop
  Vmath::Vcopy(nPts, rho, 1, rho_1, 1); // rho_1 = rho

  // Do I need to do this? TODO
  m_fields[rho_index]->FwdTrans(m_fields[rho_index]->GetPhys(),
      m_fields[rho_index]->UpdateCoeffs());
  m_fields[rho_minus_index]->FwdTrans(m_fields[rho_minus_index]->GetPhys(),
      m_fields[rho_minus_index]->UpdateCoeffs());
}

void MaxwellWaveSystem::LorenzGaugeSolve(const int field_t_index,
                                          const int field_t_minus1_index,
                                          const int source_index) {
  // copy across into shorter variable names to make sure code fits
  // on one line, more readable that way.
  const int f0     = field_t_index;
  const int f_1    = field_t_minus1_index;
  const int s      = source_index;
  const int nPts   = GetNpoints();
  const int nCfs   = GetNcoeffs();
  const double dt2 = std::pow(m_timestep, 2);

  auto f0phys  = m_fields[f0]->UpdatePhys();
  auto f_1phys = m_fields[f_1]->UpdatePhys();
  auto sphys   = m_fields[s]->GetPhys();
  auto &f0coeff  = m_fields[f0]->UpdateCoeffs();
  auto &f_1coeff = m_fields[f_1]->UpdateCoeffs();
  auto scoeff   = m_fields[s]->GetCoeffs();

  Array<OneD, NekDouble> rhs(nCfs, 0.0), tmp(nCfs, 0.0), tmp2(nCfs, 0.0);

  // Apply mass matrix op -> tmp
  MultiRegions::GlobalMatrixKey massKey(StdRegions::eMass);

  // Apply Laplacian matrix op -> tmp
  MultiRegions::GlobalMatrixKey laplacianKey(StdRegions::eLaplacian);

  if (m_theta == 0.0) {
    // Evaluate M^{-1} * L * u
    m_fields[f0]->GeneralMatrixOp(laplacianKey, f0coeff, tmp);
    m_fields[f0]->MultiplyByInvMassMatrix(tmp, tmp2);

    // Temporary copy for f_0 to transfer to f_{-1}
    Vmath::Vcopy(nCfs, f0coeff, 1, tmp, 1);

    // Central difference timestepping
    for (int i = 0; i < nCfs; ++i)
    {
        f0coeff[i] = 2 * f0coeff[i] - dt2 * tmp2[i] - f_1coeff[i] + dt2 * scoeff[i];
    }

    // Update f_{-1}
    Vmath::Vcopy(nCfs, tmp, 1, f_1coeff, 1);

    // backward transform
    m_fields[f0]->BwdTrans(f0coeff, f0phys);
    m_fields[f_1]->BwdTrans(f_1coeff, f_1phys);
  } else {
        // need in the form (∇² - lambda)f⁺ = rhs, where
    double lambda = 2.0 / dt2 / m_theta;

    for (int i = 0; i < nCfs; ++i)
    {
        // This is negative, because HelmSolve will negate the input to be
        // consistent with the Helmholtz equation definition.
        tmp2[i] = -lambda * (2 * f0coeff[i] - f_1coeff[i]);
    }

    m_fields[f0]->GeneralMatrixOp(massKey, tmp2, rhs);

    for (int i = 0; i < nCfs; ++i)
    {
        tmp2[i] = -(2 * (1 - m_theta) / m_theta * f0coeff[i] + f_1coeff[i]);
    }

    // zero tmp
    Vmath::Zero(nCfs, tmp, 1);

    // now do diffusion operator
    m_fields[f0]->GeneralMatrixOp(laplacianKey, tmp2, tmp);

    for (int i = 0; i < nCfs; ++i)
    {
        // copy the second term and sources into rhs
        // being careful to subtract rather than add because of the HelmSolve
        rhs[i] -= tmp[i] + 2.0 / m_theta * scoeff[i];
    }

    // Zero storage
    Vmath::Zero(nCfs, tmp2, 1);

    m_factors[StdRegions::eFactorLambda] = lambda;

    m_fields[f0]->HelmSolve(rhs, tmp2, m_factors, StdRegions::NullVarCoeffMap,
                            MultiRegions::NullVarFactorsMap,
                            NullNekDouble1DArray, false);

    // Rotate storage
    Vmath::Vcopy(nCfs, f0coeff, 1, f_1coeff, 1);
    Vmath::Vcopy(nCfs, tmp2, 1, f0coeff, 1);

    m_fields[f0]->BwdTrans(f0coeff, f0phys);
    m_fields[f_1]->BwdTrans(f_1coeff, f_1phys);

  }
}


} // namespace Nektar
