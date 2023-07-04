#include <neso_particles.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>

struct ioniseData {

  ioniseData() = default;

  explicit ioniseData(const double& dt_,
                      const double& t_to_SI_,
                      const double& n_to_SI_,
                      const double& k_cos_theta_,
                      const double& k_sin_theta_,
                      const double& particle_remove_key_):
  dt(dt_), 
  t_to_SI(t_to_SI_),
  n_to_SI(n_to_SI_),
  k_cos_theta(k_cos_theta_),
  k_sin_theta(k_sin_theta_),
  particle_remove_key(particle_remove_key_) {}

  const double dt, t_to_SI, n_to_SI, k_cos_theta, k_sin_theta, particle_remove_key;

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

  const INT k_remove_key = particle_remove_key;

  mutable REAL TeV, invratio, k_V_0, k_V_1, n_SI, k_SD, k_SM_0, k_SM_1, k_SE, k_W;
  mutable INT k_ID, k_internal_state;
  mutable REAL deltaweight = 0.0;
};