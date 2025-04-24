
#include <LibUtilities/TimeIntegration/TimeIntegrationScheme.h>
#include <iomanip>
#include <iostream>
#include <tinyxml.h>

#include "DiffusionSystem.hpp"
#include <boost/core/ignore_unused.hpp>

namespace MR = Nektar::MultiRegions;

namespace NESO::Solvers::Diffusion {
std::string DiffusionSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "UnsteadyDiffusion", DiffusionSystem::create);

DiffusionSystem::DiffusionSystem(const LU::SessionReaderSharedPtr &session,
                                 const SD::MeshGraphSharedPtr &graph)
    : TimeEvoEqnSysBase<SU::UnsteadySystem, Particles::EmptyPartSys>(session,
                                                                     graph) {}

/**
 * @brief Compute the projection for the unsteady diffusion problem.
 *
 * @param in_arr Unprojected field values
 * @param out_arr Projected values
 * @param time The current time
 */
void DiffusionSystem::do_ode_projection(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time) {
  SetBoundaryConditions(time);

  Array<OneD, NekDouble> coeffs(m_fields[0]->GetNcoeffs());

  for (auto i = 0; i < in_arr.size(); ++i) {
    m_fields[i]->FwdTrans(in_arr[i], coeffs);
    m_fields[i]->BwdTrans(coeffs, out_arr[i]);
  }
}

/**
 * @brief Implicit solution of the unsteady diffusion problem.
 *
 * @param in_arr Field values
 * @param out_arr RHS array
 * @param time Current time
 * @param lambda Implicit lambda factor
 */
void DiffusionSystem::do_implicit_solve(
    const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
    Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time,
    const NekDouble lambda) {
  boost::ignore_unused(time);

  // Update lambda factor
  this->helmsolve_factors[SR::eFactorLambda] = 1.0 / lambda / this->epsilon;

  /*
  We solve ( \nabla^2 - HHlambda ) Y[i] = rhs [i]
    in_arr = input: \hat{rhs} -> output: \hat{Y}
    out_arr = output: nabla^2 \hat{Y}
    where \hat = modal coeffs
  */
  for (int i = 0; i < in_arr.size(); ++i) {
    // Multiply 1.0/timestep/lambda
    Vmath::Smul(this->n_pts, -this->helmsolve_factors[SR::eFactorLambda],
                in_arr[i], 1, out_arr[i], 1);

    // Solve a system of equations with Helmholtz solver
    m_fields[i]->HelmSolve(out_arr[i], m_fields[i]->UpdateCoeffs(),
                           this->helmsolve_factors, this->helmsolve_varcoeffs);

    m_fields[i]->BwdTrans(m_fields[i]->GetCoeffs(), out_arr[i]);

    m_fields[i]->SetPhysState(false);
  }
}

/**
 * @brief Extract default collection type from the session
 * @returns a NC::ImplementationType enum describing the collection type
 */
NC::ImplementationType DiffusionSystem::get_collection_type() {
  NC::ImplementationType collection_type = NC::eNoImpType;
  TiXmlDocument &doc = m_session->GetDocument();
  TiXmlHandle doc_handle(&doc);
  TiXmlElement *root_node = doc_handle.FirstChildElement("NEKTAR").Element();
  TiXmlElement *collections_node = root_node->FirstChildElement("COLLECTIONS");
  if (collections_node) {
    const char *default_collection_type_str =
        collections_node->Attribute("DEFAULT");
    const std::string collection_type_str =
        default_collection_type_str ? std::string(default_collection_type_str)
                                    : "NoImplementationType";
    for (int i = 0; i < NC::SIZE_ImplementationType; ++i) {
      if (collection_type_str == NC::ImplementationTypeMap[i]) {
        collection_type = (NC::ImplementationType)i;
        break;
      }
    }
  }
  return collection_type;
}

/**
 * @brief Read configuration options from file.
 *
 */
void DiffusionSystem::load_params() {
  m_session->MatchSolverInfo("SpectralVanishingViscosity", "True",
                             this->use_spec_van_visc, false);

  if (this->use_spec_van_visc) {
    m_session->LoadParameter("SVVCutoffRatio", this->sVV_cutoff_ratio, 0.75);
    m_session->LoadParameter("SVVDiffCoeff", this->sVV_diff_coeff, 0.1);
  }

  m_session->LoadParameter("epsilon", this->epsilon, 1.0);
  m_session->LoadParameter("k_par", this->k_par, 100.0);
  m_session->LoadParameter("k_perp", this->k_perp, 1.0);
  m_session->LoadParameter("n", this->n, 1e18);
  // Theta is read in degrees; convert to radians.
  m_session->LoadParameter("theta", this->theta, 0.0);
  this->theta *= -M_PI / 180.0;
}

/**
 * @brief Populate the factors and coefficients used in the Helmsolve() call,
 * including the values of the diffusion tensor.
 *
 */
void DiffusionSystem::setup_helmsolve_coeffs() {
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

  /* OP: Think this was put in because variable coefficients weren't supported
   * with MatrixFree collections; might not be needed with newer Nektar
   * versions?
   */
  //===== Special behaviour for MatrixFree collections =====
  switch (get_collection_type()) {
  case NC::ImplementationType::eMatrixFree:
    // For matrix-free collections set up constant coefficients
    this->helmsolve_factors[SR::eFactorCoeffD00] = d00;
    this->helmsolve_factors[SR::eFactorCoeffD01] = d01;
    this->helmsolve_factors[SR::eFactorCoeffD11] = d11;
    break;
  default:
    // For all other implementations, or if no collection was specified, set up
    // variable coefficients
    this->helmsolve_varcoeffs[SR::eVarCoeffD00] =
        Array<OneD, NekDouble>(this->n_pts, d00);
    this->helmsolve_varcoeffs[SR::eVarCoeffD01] =
        Array<OneD, NekDouble>(this->n_pts, d01);
    this->helmsolve_varcoeffs[SR::eVarCoeffD11] =
        Array<OneD, NekDouble>(this->n_pts, d11);
    break;
  }
  //===== End of special behaviour for MatrixFree collections =====
}

/**
 * @brief Output configuration options associated with this equation system.
 *
 * @param s A Nektar++ SummaryList object
 */
void DiffusionSystem::v_GenerateSummary(SU::SummaryList &s) {
  UnsteadySystem::v_GenerateSummary(s);
  SU::AddSummaryItem(s, "epsilon", this->epsilon);
  SU::AddSummaryItem(s, "k_par", this->k_par);
  SU::AddSummaryItem(s, "k_perp", this->k_perp);
  SU::AddSummaryItem(s, "n", this->n);
  SU::AddSummaryItem(s, "theta (rads)", this->theta);
  if (this->use_spec_van_visc) {
    std::stringstream ss;
    ss << "SVV (cut off = " << this->sVV_cutoff_ratio
       << ", coeff = " << this->sVV_diff_coeff << ")";
    SU::AddSummaryItem(s, "Smoothing", ss.str());
  }
}

/**
 * @brief Initialisation object for the unsteady diffusion problem.
 */
void DiffusionSystem::v_InitObject(bool DeclareField) {
  TimeEvoEqnSysBase::v_InitObject(DeclareField);

  // CG only
  ASSERTL0(m_projectionType == MR::eGalerkin,
           "Only continuous Galerkin discretisation supported.");

  m_ode.DefineImplicitSolve(&DiffusionSystem::do_implicit_solve, this);

  setup_helmsolve_coeffs();
}

} // namespace NESO::Solvers::Diffusion
