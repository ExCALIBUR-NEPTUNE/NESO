#include "particle_group.hpp"
#include "typedefs.hpp"
#include <cstddef>
#include <future>
#include <neso_particles.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>


using namespace NESO::Particles;

struct baseReactionData{
    public:
      explicit baseReactionData(
      const double& dt_,
      const double& t_to_SI_,
      const double& n_to_SI_,
      const double& k_cos_theta_,
      const double& k_sin_theta_,
      const ParticleGroupSharedPtr& particle_group_
    ):
      dt(dt_), 
      t_to_SI(t_to_SI_),
      n_to_SI(n_to_SI_),
      k_cos_theta(k_cos_theta_),
      k_sin_theta(k_sin_theta_),
      particle_group(particle_group_)
    {
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

    const double dt, t_to_SI, n_to_SI, k_cos_theta, k_sin_theta;

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

};

struct ioniseData: public baseReactionData {

  ioniseData(
    const double& dt_,
    const double& t_to_SI_,
    const double& n_to_SI_,
    const double& k_cos_theta_,
    const double& k_sin_theta_,
    const ParticleGroupSharedPtr& particle_group_
  ): baseReactionData(
    dt_,
    t_to_SI_,
    n_to_SI_,
    k_cos_theta_,
    k_sin_theta_,
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
};