#include <neso_particles.hpp>
#include <NEC_abstract_reaction/reaction_data.hpp>

using namespace NESO::Particles;

// TODO move this to the correct place
/**
 * Evaluate the Barry et al approximation to the exponential integral function
 * https://en.wikipedia.org/wiki/Exponential_integral E_1(x)
 */
inline double expint_barry_approx(const double x) {
  constexpr double gamma_Euler_Mascheroni = 0.5772156649015329;
  const double G = std::exp(-gamma_Euler_Mascheroni);
  const double b = std::sqrt(2 * (1 - G) / G / (2 - G));
  const double h_inf = (1 - G) * (std::pow(G, 2) - 6 * G + 12) /
                       (3 * G * std::pow(2 - G, 2) * b);
  const double q = 20.0 / 47.0 * std::pow(x, std::sqrt(31.0 / 26.0));
  const double h = 1 / (1 + x * std::sqrt(x)) + h_inf * q / (1 + q);
  const double logfactor =
      std::log(1 + G / x - (1 - G) / std::pow(h + b * x, 2));
  return std::exp(-x) / (G + (1 - G) * std::exp(-(x / (1 - G)))) * logfactor;
}

template <typename derived_reaction>

struct base_reaction {

  base_reaction() = default;

  base_reaction(
    const std::vector<INT> &in_states_, const std::vector<INT> &out_states_
  ): in_states(in_states_), out_states(out_states_) {
    set_test_states();
  }

  void set_test_states() const {
    std::vector<int> in_test_states_;
    std::vector<int> out_test_states_;

    for (int state_index = 0; state_index < in_states.size(); state_index++) {
      if (in_states[state_index] > 0) {
        in_test_states_.push_back(state_index);
      }
    }

    this->in_test_states = in_test_states_;

    for (int state_index = 0; state_index < out_states.size(); state_index++) {
      if (out_states[state_index] > 0) {
        out_test_states_.push_back(state_index);
      }
    }

    this->out_test_states = out_test_states_;
  }

  REAL calc_rate(const ioniseData& reactionData) const {
    const auto& underlying = static_cast<const derived_reaction&>(*this);

    return underlying.template calcRate(reactionData);
  }

  std::vector<REAL> scattering_kernel() const {
    const auto& underlying = static_cast<const derived_reaction&>(*this);

    return underlying.template scattering_kernel();
  }

  void feedback_kernel(ioniseData& reactionData, REAL& weight_fraction) {
    const auto& underlying = static_cast<const derived_reaction&>(*this);

    return underlying.template feedback_kernel(reactionData, weight_fraction);
  }

  void transformation_kernel() {    
    for (auto& out_test_state : this->out_test_states) {
      this->post_collision_internal_states.push_back(0);
    }
  }

  void weight_kernel() {
    for (auto& out_test_state : this->out_test_states) {
      this->post_collision_weights.push_back(0.0);
    }
  }

  void apply_kernel() const {
    const auto& underlying = static_cast<const derived_reaction&>(*this);

    return underlying.template apply_kernel();
  }

  protected:
    const std::vector<INT> in_states, out_states;
    mutable std::vector<int> in_test_states, out_test_states;
    std::vector<INT> post_collision_internal_states{};
    std::vector<REAL> post_collision_weights{};
};

struct ionise_reaction : public base_reaction<ionise_reaction> {

  ionise_reaction() = default;

  explicit ionise_reaction(
    const std::vector<INT> &in_states_, const std::vector<INT> &out_states_
  ) : base_reaction(in_states_, out_states_) {}

  REAL calc_rate(ioniseData* reactionDataPtr) const {
    const REAL TeV = reactionDataPtr->real_fields[0];
    const REAL invratio = reactionDataPtr->invratio;
    const double k_rate_factor = reactionDataPtr->k_rate_factor;
    const double k_b_i_expc_i = reactionDataPtr->k_b_i_expc_i;
    const double k_c_i = reactionDataPtr->k_c_i;

    const REAL rate = -k_rate_factor / (TeV * std::sqrt(TeV)) *
                      (expint_barry_approx(invratio) / invratio +
                      (k_b_i_expc_i / (invratio + k_c_i)) *
                      expint_barry_approx(invratio + k_c_i));
  
    return rate;
  }

  std::vector<REAL> scattering_kernel() const {
    std::vector<REAL> post_collision_velocities;
    
    for (auto& out_test_state : out_test_states) {
      post_collision_velocities.push_back(0.0);
    }
    
    return post_collision_velocities;
  }

  void feedback_kernel(ioniseData* reactionDataPtr, REAL& weight_fraction) {
    const double k_cos_theta = std::cos(reactionDataPtr->theta);
    const double k_sin_theta = std::sin(reactionDataPtr->theta);
    REAL k_SD = reactionDataPtr->real_fields[3];
    const REAL k_V_0 = reactionDataPtr->particle_properties_2d_x[0];
    const REAL k_V_1 = reactionDataPtr->particle_properties_2d_y[0];
    REAL k_SM_0 = reactionDataPtr->fields_2d_x[0];
    REAL k_SM_1 = reactionDataPtr->fields_2d_y[0];
    REAL k_SE = reactionDataPtr->real_fields[2];
    const double k_n_scale = reactionDataPtr->k_n_scale;
    const double inv_k_dt = reactionDataPtr->inv_k_dt;

    k_SD = -reactionDataPtr->particle_properties[0] * weight_fraction * k_n_scale * inv_k_dt;

    const REAL v_s = k_V_0 * k_cos_theta + k_V_1 * k_sin_theta;

    k_SM_0 = k_SD * v_s * k_cos_theta;
    k_SM_1 = k_SD * v_s * k_sin_theta;

    k_SE = k_SD * v_s * v_s / 2;

    reactionDataPtr->real_fields[3] = k_SD;
    reactionDataPtr->fields_2d_x[0] = k_SM_0;
    reactionDataPtr->fields_2d_y[0] = k_SM_1;
    reactionDataPtr->real_fields[2] = k_SE;
  }

  void apply_kernel() {
    scattering_kernel();
    weight_kernel();
    transformation_kernel();
  }
};