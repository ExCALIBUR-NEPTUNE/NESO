#include <boost/core/ignore_unused.hpp>

#include "SourceTerms.hpp"

namespace NESO::Solvers::SimpleSOL {
std::string SourceTerms::class_name =
    SU::GetForcingFactory().RegisterCreatorFunction(
        "SourceTerms", SourceTerms::create, "Source terms for 1D SOL code");

SourceTerms::SourceTerms(const LU::SessionReaderSharedPtr &session,
                         const std::weak_ptr<SU::EquationSystem> &equation_sys)
    : Forcing(session, equation_sys), field_to_index(session->GetVariables()) {}

void SourceTerms::v_InitObject(const Array<OneD, MR::ExpListSharedPtr> &fields,
                               const unsigned int &num_src_fields,
                               const TiXmlElement *force_xml_node) {
  boost::ignore_unused(force_xml_node);

  // smax should be determined from max(this->s) for all tasks... just set it
  // via a parameter for now.
  m_session->LoadParameter("unrotated_x_max", this->smax, 110.0);

  // Angle in radians between source orientation and the x-axis
  m_session->LoadParameter("theta", this->theta, 0.0);

  // Width of sources
  m_session->LoadParameter("srcs_sigma", this->sigma, 2.0);

  double source_mask;
  m_session->LoadParameter("srcs_mask", source_mask, 1.0);

  int spacedim = fields[0]->GetGraph()->GetSpaceDimension();
  int num_pts = fields[0]->GetTotPoints();

  m_NumVariable = num_src_fields;

  // Compute s - coord parallel to source term orientation
  Array<OneD, NekDouble> tmp_x = Array<OneD, NekDouble>(num_pts);
  Array<OneD, NekDouble> tmp_y = Array<OneD, NekDouble>(num_pts);
  this->s = Array<OneD, NekDouble>(num_pts);
  fields[0]->GetCoords(tmp_x, tmp_y);
  for (auto ii = 0; ii < num_pts; ii++) {
    this->s[ii] = tmp_x[ii] * cos(this->theta) + tmp_y[ii] * sin(this->theta);
  }

  //===== Set up source term constants from session =====
  // Source term normalisation is calculated relative to the sigma=2 case
  constexpr NekDouble sigma0 = 2.0;

  // (Gaussian sources always positioned halfway along s dimension)
  this->mu = this->smax / 2;

  // Set normalisation factors for the chosen sigma
  this->rho_prefac =
      source_mask * 3.989422804e-22 * 1e21 * sigma0 / this->sigma;
  this->u_prefac = source_mask * 7.296657414e-27 * -1e26 * sigma0 / this->sigma;
  this->E_prefac =
      source_mask * 7.978845608e-5 * 30000.0 * sigma0 / this->sigma;
}

NekDouble calc_gaussian(NekDouble prefac, NekDouble mu, NekDouble sigma,
                        NekDouble s) {
  return prefac * exp(-(mu - s) * (mu - s) / 2 / sigma / sigma);
}

void SourceTerms::v_Apply(const Array<OneD, MR::ExpListSharedPtr> &fields,
                          const Array<OneD, Array<OneD, NekDouble>> &in_arr,
                          Array<OneD, Array<OneD, NekDouble>> &out_arr,
                          const NekDouble &time) {
  boost::ignore_unused(time);
  unsigned short ndims = fields[0]->GetGraph()->GetSpaceDimension();

  int rho_idx = this->field_to_index.get_idx("rho");
  int rhou_idx = this->field_to_index.get_idx("rhou");
  int rhov_idx = this->field_to_index.get_idx("rhov");
  int E_idx = this->field_to_index.get_idx("E");

  // Density source term
  for (int i = 0; i < out_arr[rho_idx].size(); ++i) {
    out_arr[rho_idx][i] +=
        calc_gaussian(this->rho_prefac, this->mu, this->sigma, this->s[i]);
  }
  // rho*u source term
  for (int i = 0; i < out_arr[rhou_idx].size(); ++i) {
    out_arr[rhou_idx][i] +=
        std::cos(this->theta) * (this->s[i] / this->mu - 1.) *
        calc_gaussian(this->u_prefac, this->mu, this->sigma, this->s[i]);
  }
  if (ndims == 2) {
    // rho*v source term
    for (int i = 0; i < out_arr[rhov_idx].size(); ++i) {
      out_arr[rhov_idx][i] +=
          std::sin(this->theta) * (this->s[i] / this->mu - 1.) *
          calc_gaussian(this->u_prefac, this->mu, this->sigma, this->s[i]);
    }
  }

  // E source term - divided by 2 since the LHS of the energy equation has
  // been doubled
  for (int i = 0; i < out_arr[E_idx].size(); ++i) {
    out_arr[E_idx][i] +=
        calc_gaussian(this->E_prefac, this->mu, this->sigma, this->s[i]) / 2.0;
  }

  // Add sources stored as separate fields, if they exist
  std::vector<std::string> target_fields = {"rho", "rhou", "rhov", "E"};
  for (auto target_field : target_fields) {
    int src_field_idx = this->field_to_index.get_idx(target_field + "_src");
    if (src_field_idx >= 0) {
      int dst_field_idx = this->field_to_index.get_idx(target_field);
      if (dst_field_idx >= 0) {
        auto phys_vals = fields[src_field_idx]->GetPhys();
        for (int i = 0; i < out_arr[dst_field_idx].size(); ++i) {
          out_arr[dst_field_idx][i] += phys_vals[i];
        }
      }
    }
  }
}

} // namespace NESO::Solvers::SimpleSOL
