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
    const std::vector<INT> &in_states_,
    const std::vector<INT> &out_states_,
    const double& dt_,
    const double& t_to_SI_,
    const double& n_to_SI_,
    const double& theta_,
    const ParticleGroupSharedPtr& particle_group_
  ): in_states(in_states_),
     out_states(out_states_),
     dt(dt_), 
    t_to_SI(t_to_SI_),
    n_to_SI(n_to_SI_),
    theta(theta_),
    particle_group(particle_group_) {
    set_test_states();

    for (std::size_t i = 0; i < required_int_fields.size(); i++) {
      std::string int_field_ID = available_int_fields[required_int_fields[i]];
      int_fields_buffers[i] = (*particle_group)[Sym<INT>(int_field_ID)]->cell_dat.device_ptr();
    }

    for (std::size_t i = 0; i < required_real_fields.size(); i++) {
      std::string real_field_ID = available_real_fields[required_real_fields[i]];
      real_fields_buffers[i] = (*particle_group)[Sym<REAL>(real_field_ID)]->cell_dat.device_ptr();
    }

    for (std::size_t i = 0; i < required_particle_properties.size(); i++) {
      std::string real_field_ID = available_particle_properties[required_particle_properties[i]];
      particle_properties_buffers[i] = (*particle_group)[Sym<REAL>(real_field_ID)]->cell_dat.device_ptr();
    }

    for (std::size_t i = 0; i < required_2d_fields.size(); i++) {
      std::string field_2d_ID = available_real_fields[required_2d_fields[i]];
      fields_2d_buffers[i] = (*particle_group)[Sym<REAL>(field_2d_ID)]->cell_dat.device_ptr();
    }

    for (std::size_t i = 0; i < required_2d_particle_properties.size(); i++) {
      std::string property_2d_ID = available_particle_properties[required_2d_particle_properties[i]];
      particle_properties_2d_buffers[i] = (*particle_group)[Sym<REAL>(property_2d_ID)]->cell_dat.device_ptr();
    }
  }

  void select_params(const INT& cellx, const INT& layerx) {
    
      for (std::size_t i = 0; i < required_int_fields.size(); i++) {
        int_fields[i] = int_fields_buffers[i][cellx][0][layerx];
      }

      for (std::size_t i = 0; i < required_real_fields.size(); i++) {
        real_fields[i] = real_fields_buffers[i][cellx][0][layerx];
      }

      for (std::size_t i = 0; i < required_particle_properties.size(); i++) {
        particle_properties[i] = particle_properties_buffers[i][cellx][0][layerx];
      }

      for (std::size_t i = 0; i < required_2d_fields.size(); i++) {
        fields_2d_x[i] = fields_2d_buffers[i][cellx][0][layerx];
        fields_2d_y[i] = fields_2d_buffers[i][cellx][1][layerx];
      }

      for (std::size_t i = 0; i < required_2d_particle_properties.size(); i++) {
        particle_properties_2d_x[i] = particle_properties_2d_buffers[i][cellx][0][layerx];
        particle_properties_2d_y[i] = particle_properties_2d_buffers[i][cellx][1][layerx];
      }

    }

  void update_params(const INT& cellx, const INT& layerx) {

    for (std::size_t i = 0; i < required_int_fields.size(); i++) {
      int_fields_buffers[i][cellx][0][layerx] = int_fields[i];
    }

    for (std::size_t i = 0; i < required_real_fields.size(); i++) {
      real_fields_buffers[i][cellx][0][layerx] = real_fields[i];
    }

    for (std::size_t i = 0; i < required_particle_properties.size(); i++) {
      particle_properties_buffers[i][cellx][0][layerx] = particle_properties[i];
    }

    for (std::size_t i = 0; i < required_2d_fields.size(); i++) {
      fields_2d_buffers[i][cellx][0][layerx] = fields_2d_x[i];
      fields_2d_buffers[i][cellx][1][layerx] = fields_2d_y[i];
    }

    for (std::size_t i = 0; i < required_2d_particle_properties.size(); i++) {
      particle_properties_2d_buffers[i][cellx][0][layerx] = particle_properties_2d_x[i];
      particle_properties_2d_buffers[i][cellx][1][layerx] = particle_properties_2d_y[i];
    }
  }

  const double dt, t_to_SI, n_to_SI, theta;

  ParticleGroupSharedPtr particle_group;

  REAL deltaweight = 0.0;
  const std::vector<std::string> available_int_fields{
    "CELL_ID", // INT
    "PARTICLE_ID", // INT
    "INTERNAL_STATE" // INT
  };
  const std::vector<std::string> available_real_fields{
    "SOURCE_DENSITY", // REAL
    "SOURCE_MOMENTUM", // REAL
    "SOURCE_ENERGY", // REAL
    "ELECTRON_DENSITY", // REAL
    "ELECTRON_TEMPERATURE" // REAL
  };
  const std::vector<std::string> available_particle_properties{
    "COMPUTATIONAL_WEIGHT", // REAL
    "MASS", // REAL
    "VELOCITY", // REAL
    "POSITION" // REAL
  };

  static constexpr std::array<int, 2> required_int_fields{1, 2};
  static constexpr std::array<int, 4> required_real_fields{4, 3, 2, 0};
  static constexpr std::array<int, 1> required_particle_properties{0};
  static constexpr std::array<int, 1> required_2d_fields{1};
  static constexpr std::array<int, 1> required_2d_particle_properties{2};

  std::array<INT***, required_int_fields.size()> int_fields_buffers;
  std::array<REAL***, required_real_fields.size()> real_fields_buffers;
  std::array<REAL***, required_particle_properties.size()> particle_properties_buffers;
  std::array<REAL***, required_2d_fields.size()> fields_2d_buffers;
  std::array<REAL***, required_2d_particle_properties.size()> particle_properties_2d_buffers;

  std::array<INT, required_int_fields.size()> int_fields;
  std::array<REAL, required_real_fields.size()> real_fields;
  std::array<REAL, required_particle_properties.size()> particle_properties;
  std::array<REAL, required_2d_fields.size()> fields_2d_x;
  std::array<REAL, required_2d_fields.size()> fields_2d_y;
  std::array<REAL, required_2d_particle_properties.size()> particle_properties_2d_x;
  std::array<REAL, required_2d_particle_properties.size()> particle_properties_2d_y;

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

  std::vector<REAL> scattering_kernel(REAL& weight_fraction) const {
    const auto& underlying = static_cast<const derived_reaction&>(*this);

    return underlying.template scattering_kernel(weight_fraction);
  }

  void feedback_kernel(ioniseData& reactionData, REAL& weight_fraction) {
    const auto& underlying = static_cast<const derived_reaction&>(*this);

    return underlying.template feedback_kernel(reactionData, weight_fraction);
  }

  void transformation_kernel(REAL& weight_fraction) {    
    for (auto& out_test_state : this->out_test_states) {
      this->post_collision_internal_states.push_back(0);
    }
  }

  void weight_kernel(REAL& weight_fraction) {
    for (auto& out_test_state : this->out_test_states) {
      this->post_collision_weights.push_back(0.0);
    }
  }

  void apply_kernel(REAL& weight_fraction) const {
    const auto& underlying = static_cast<const derived_reaction&>(*this);

    return underlying.template apply_kernel(weight_fraction);
  }

  protected:
    const std::vector<INT> in_states, out_states;
    mutable std::vector<int> in_test_states, out_test_states;
    std::vector<INT> post_collision_internal_states{};
    std::vector<REAL> post_collision_weights{};
};

struct ionise_reaction : public base_reaction<ionise_reaction> {

  public: ionise_reaction() = delete;

  explicit ionise_reaction(
    const std::vector<INT> &in_states_,
    const std::vector<INT> &out_states_,
    const double& dt_,
    const double& t_to_SI_,
    const double& n_to_SI_,
    const double& theta_,
    const ParticleGroupSharedPtr& particle_group_
  ) : base_reaction(
    in_states_,
    out_states_,
    dt_, 
    t_to_SI_,
    n_to_SI_,
    theta_,
    particle_group_
  ) {}

  const double k_dt = dt;
  const double inv_k_dt = 1 / k_dt;
  const double k_dt_SI = dt * t_to_SI;
  const double k_n_scale = 1 / n_to_SI;

  const double k_a_i = 4.0e-14; // a_i constant for hydrogen (a_1)
  const double k_b_i = 0.6;     // b_i constant for hydrogen (b_1)
  const double k_c_i = 0.56;    // c_i constant for hydrogen (c_1)
  const double k_E_i =
      13.6; // E_i binding energy for most bound electron in hydrogen (E_1)
  const double k_q_i = 1.0; // Number of electrons in inner shell for hydrogen
  const double k_b_i_expc_i =
      k_b_i * std::exp(k_c_i); // exp(c_i) constant for hydrogen (c_1)

  const double k_rate_factor =
      -k_q_i * 6.7e7 * k_a_i * 1e-6; // 1e-6 to go from cm^3 to m^3

  void set_invratio() {
    invratio = k_E_i / real_fields[0];
  }

  REAL invratio;

  REAL calc_rate(ionise_reaction* reactionDataPtr) const {
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

  std::vector<REAL> scattering_kernel(REAL& weight_fraction) const {
    std::vector<REAL> post_collision_velocities;
    
    for (auto& out_test_state : out_test_states) {
      post_collision_velocities.push_back(0.0);
    }
    
    return post_collision_velocities;
  }

  void feedback_kernel(ionise_reaction* reactionDataPtr, REAL& weight_fraction) {
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

  void apply_kernel(REAL& weight_fraction) {
    scattering_kernel(weight_fraction);
    weight_kernel(weight_fraction);
    transformation_kernel(weight_fraction);
  }
};