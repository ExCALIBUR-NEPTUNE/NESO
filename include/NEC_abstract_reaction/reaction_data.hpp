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

    baseReactionData() = default;

    void select_params(const INT& cellx, const INT& layerx) const{
      k_TeV_i = k_TeV[cellx][0][layerx];
      k_W_i = k_W[cellx][0][layerx];
      n_SI = k_n[cellx][0][layerx];
      k_SD_i = k_SD[cellx][0][layerx];
      k_SM_0 = k_SM[cellx][0][layerx];
      k_SM_1 = k_SM[cellx][1][layerx];
      k_SE_i = k_SE[cellx][0][layerx];
      k_V_0 = k_V[cellx][0][layerx];
      k_V_1 = k_V[cellx][1][layerx];
      k_ID_i = k_ID[cellx][0][layerx];
      k_internal_state_i = k_internal_state[cellx][0][layerx];
    }

    void update_params(const INT& cellx, const INT& layerx) const {
      k_TeV[cellx][0][layerx] = k_TeV_i;
      k_W[cellx][0][layerx] = k_W_i;
      k_n[cellx][0][layerx] = n_SI;
      k_SD[cellx][0][layerx] = k_SD_i;
      k_SM[cellx][0][layerx] = k_SM_0;
      k_SM[cellx][1][layerx] = k_SM_1;
      k_SE[cellx][0][layerx] = k_SE_i;
      k_V[cellx][0][layerx] = k_V_0;
      k_V[cellx][1][layerx] = k_V_1;
      k_ID[cellx][0][layerx] = k_ID_i;
      k_internal_state[cellx][0][layerx] = k_internal_state_i;
    }

    public:
      double ***k_TeV, ***k_W, ***k_n, ***k_SD, ***k_SE, ***k_SM, ***k_V;
      long ***k_ID, ***k_internal_state;
      mutable REAL k_TeV_i, k_V_0, k_V_1, n_SI, k_SD_i, k_SM_0, k_SM_1, k_SE_i, k_W_i;
      mutable INT k_ID_i, k_internal_state_i;
      mutable REAL deltaweight = 0.0;

};

struct ioniseData: public baseReactionData {

  explicit ioniseData(const double& dt_,
                      const double& t_to_SI_,
                      const double& n_to_SI_,
                      const double& k_cos_theta_,
                      const double& k_sin_theta_,
                      double ***k_TeV_,
                      double ***k_n_,
                      double ***k_SD_,
                      double ***k_SE_,
                      double ***k_SM_,
                      double ***k_V_,
                      double ***k_W_,
                      long ***k_ID_,
                      long ***k_internal_state_
                      ):
                        dt(dt_), 
                        t_to_SI(t_to_SI_),
                        n_to_SI(n_to_SI_),
                        k_cos_theta(k_cos_theta_),
                        k_sin_theta(k_sin_theta_),
                        baseReactionData() {
                          k_TeV = k_TeV_;
                          k_n = k_n_;
                          k_SD = k_SD_;
                          k_SE = k_SE_;
                          k_SM = k_SM_;
                          k_V = k_V_;
                          k_W = k_W_;
                          k_ID = k_ID_;
                          k_internal_state = k_internal_state_;
                        }

  const double dt, t_to_SI, n_to_SI, k_cos_theta, k_sin_theta;

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

  void set_invratio() const {
    invratio = k_E_i / k_TeV_i;
  }

  mutable REAL invratio;
};