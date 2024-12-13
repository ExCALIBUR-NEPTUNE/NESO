
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <iomanip>
#include <iostream>
#include <tinyxml.h>

#include "DiffusionSystem.hpp"
#include <boost/core/ignore_unused.hpp>
namespace NESO::Solvers::Diffusion {
std::string DiffusionSystem::className =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "UnsteadyDiffusion", DiffusionSystem::create);

DiffusionSystem::DiffusionSystem(const LU::SessionReaderSharedPtr &pSession,
                                 const SD::MeshGraphSharedPtr &pGraph)
    : UnsteadySystem(pSession, pGraph) {}

/**
 * @brief Initialisation object for the unsteady diffusion problem.
 */
void DiffusionSystem::v_InitObject(bool DeclareField) {
  UnsteadySystem::v_InitObject(DeclareField);

  m_session->MatchSolverInfo("SpectralVanishingViscosity", "True",
                             this->use_spec_van_visc, false);

  if (this->use_spec_van_visc) {
    m_session->LoadParameter("SVVCutoffRatio", this->sVV_cutoff_ratio, 0.75);
    m_session->LoadParameter("SVVDiffCoeff", this->sVV_diff_coeff, 0.1);
  }

  int npoints = m_fields[0]->GetNpoints();

  m_session->LoadParameter("k_par", this->k_par, 100.0);
  m_session->LoadParameter("k_perp", this->k_perp, 1.0);
  m_session->LoadParameter("theta", this->theta, 0.0);
  m_session->LoadParameter("n", this->n, 1e18);
  m_session->LoadParameter("epsilon", this->epsilon, 1.0);

  // Convert to radians.
  this->theta *= -M_PI / 180.0;

  Array<OneD, NekDouble> xc(npoints), yc(npoints);
  m_fields[0]->GetCoords(xc, yc);

  if (this->use_spec_van_visc) {
    this->helmsolve_factors[SR::eFactorSVVCutoffRatio] = this->sVV_cutoff_ratio;
    this->helmsolve_factors[SR::eFactorSVVDiffCoeff] =
        this->sVV_diff_coeff / this->epsilon;
  }

  NekDouble ct = cos(this->theta), st = sin(this->theta);
  NekDouble d00 = (2.0 / (3.0 * this->n)) *
                  ((this->k_par - this->k_perp) * ct * ct + this->k_perp);
  NekDouble d01 =
      (2.0 / (3.0 * this->n)) * ((this->k_par - this->k_perp) * ct * st);
  NekDouble d11 = (2.0 / (3.0 * this->n)) *
                  ((this->k_par - this->k_perp) * st * st + this->k_perp);

  TiXmlDocument &doc = m_session->GetDocument();
  TiXmlHandle docHandle(&doc);
  TiXmlElement *master = docHandle.FirstChildElement("NEKTAR").Element();
  TiXmlElement *xmlCol = master->FirstChildElement("COLLECTIONS");

  // Check if user has specified some options
  if (xmlCol) {
    const char *defaultImpl = xmlCol->Attribute("DEFAULT");
    const std::string collinfo = defaultImpl ? std::string(defaultImpl) : "";
    if (collinfo != "MatrixFree") {
      int nq = m_fields[0]->GetNpoints();
      // Set up variable coefficients
      this->helmsolve_varcoeffs[SR::eVarCoeffD00] =
          Array<OneD, NekDouble>(nq, d00);
      this->helmsolve_varcoeffs[SR::eVarCoeffD01] =
          Array<OneD, NekDouble>(nq, d01);
      this->helmsolve_varcoeffs[SR::eVarCoeffD11] =
          Array<OneD, NekDouble>(nq, d11);
    } else {
      // Set up constant coefficients
      this->helmsolve_factors[SR::eFactorCoeffD00] = d00;
      this->helmsolve_factors[SR::eFactorCoeffD01] = d01;
      this->helmsolve_factors[SR::eFactorCoeffD11] = d11;
    }
  } else {
    int nq = m_fields[0]->GetNpoints();
    // Set up variable coefficients
    this->helmsolve_varcoeffs[SR::eVarCoeffD00] =
        Array<OneD, NekDouble>(nq, d00);
    this->helmsolve_varcoeffs[SR::eVarCoeffD01] =
        Array<OneD, NekDouble>(nq, d01);
    this->helmsolve_varcoeffs[SR::eVarCoeffD11] =
        Array<OneD, NekDouble>(nq, d11);
  }

  ASSERTL0(m_projectionType == MultiRegions::eGalerkin,
           "Only continuous Galerkin discretisation supported.");

  m_ode.DefineImplicitSolve(&DiffusionSystem::do_implicit_solve, this);
}

/**
 * @brief Unsteady diffusion problem destructor.
 */
DiffusionSystem::~DiffusionSystem() {}

void DiffusionSystem::v_GenerateSummary(SU::SummaryList &s) {
  UnsteadySystem::v_GenerateSummary(s);
  if (this->use_spec_van_visc) {
    std::stringstream ss;
    ss << "SVV (cut off = " << this->sVV_cutoff_ratio
       << ", coeff = " << this->sVV_diff_coeff << ")";
    SU::AddSummaryItem(s, "Smoothing", ss.str());
  }
}

/**
 * @brief Compute the projection for the unsteady diffusion problem.
 *
 * @param inarray    Given fields.
 * @param outarray   Calculated solution.
 * @param time       Time.
 */
void DiffusionSystem::do_ode_projection(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time) {
  int i;
  int nvariables = inarray.size();
  SetBoundaryConditions(time);

  Array<OneD, NekDouble> coeffs(m_fields[0]->GetNcoeffs());

  for (i = 0; i < nvariables; ++i) {
    m_fields[i]->FwdTrans(inarray[i], coeffs);
    m_fields[i]->BwdTrans(coeffs, outarray[i]);
  }
}

/**
 * @brief Implicit solution of the unsteady diffusion problem.
 */
void DiffusionSystem::do_implicit_solve(
    const Array<OneD, const Array<OneD, NekDouble>> &inarray,
    Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time,
    const NekDouble lambda) {
  boost::ignore_unused(time);

  int nvariables = inarray.size();
  int npoints = m_fields[0]->GetNpoints();
  this->helmsolve_factors[SR::eFactorLambda] = 1.0 / lambda / this->epsilon;

  if (this->use_spec_van_visc) {
    this->helmsolve_factors[SR::eFactorSVVCutoffRatio] = this->sVV_cutoff_ratio;
    this->helmsolve_factors[SR::eFactorSVVDiffCoeff] =
        this->sVV_diff_coeff / this->epsilon;
  }

  // We solve ( \nabla^2 - HHlambda ) Y[i] = rhs [i]
  // inarray = input: \hat{rhs} -> output: \hat{Y}
  // outarray = output: nabla^2 \hat{Y}
  // where \hat = modal coeffs
  for (int i = 0; i < nvariables; ++i) {
    // Multiply 1.0/timestep/lambda
    Vmath::Smul(npoints, -this->helmsolve_factors[SR::eFactorLambda],
                inarray[i], 1, outarray[i], 1);

    // Solve a system of equations with Helmholtz solver
    m_fields[i]->HelmSolve(outarray[i], m_fields[i]->UpdateCoeffs(),
                           this->helmsolve_factors, this->helmsolve_varcoeffs);

    m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(), outarray[i]);

    m_fields[i]->SetPhysState(false);
  }
}
} // namespace NESO::Solvers::Diffusion
