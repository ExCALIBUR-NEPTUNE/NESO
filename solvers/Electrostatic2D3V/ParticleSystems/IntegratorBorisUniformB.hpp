#ifndef __NESOSOLVERS_ELECTROSTATIC2D3V_BORISINTEGRATOR_HPP__
#define __NESOSOLVERS_ELECTROSTATIC2D3V_BORISINTEGRATOR_HPP__

#include <neso_particles.hpp>

namespace NP = NESO::Particles;

#ifndef ELEC_PIC_2D3V_CROSS_PRODUCT_3D
#define ELEC_PIC_2D3V_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)     \
  (c1) = ((a2) * (b3)) - ((a3) * (b2));                                        \
  (c2) = ((a3) * (b1)) - ((a1) * (b3));                                        \
  (c3) = ((a1) * (b2)) - ((a2) * (b1));
#endif

namespace NESO::Solvers::Electrostatic2D3V {

class IntegratorBorisUniformB {

private:
  NP::ParticleGroupSharedPtr particle_group;
  NP::SYCLTargetSharedPtr sycl_target;

  double dt;
  double B_0;
  double B_1;
  double B_2;
  double particle_E_coefficient;

public:
  /**
   *  Set the constant and uniform magnetic field over the entire domain.
   *
   *  @param B0 Magnetic fiel B in x direction.
   *  @param B1 Magnetic fiel B in y direction.
   *  @param B2 Magnetic fiel B in z direction.
   */
  inline void set_B_field(const NP::REAL B0 = 0.0, const NP::REAL B1 = 0.0,
                          const NP::REAL B2 = 0.0) {
    this->B_0 = B0;
    this->B_1 = B1;
    this->B_2 = B2;
  }

  /**
   *  Set a scaling coefficient x such that the effect of the electric field is
   *  xqE instead of qE.
   *
   *  @param x New scaling coefficient.
   */
  inline void set_E_coefficent(const NP::REAL x) {
    this->particle_E_coefficient = x;
  }

  IntegratorBorisUniformB(NP::ParticleGroupSharedPtr particle_group, double &dt,
                          double &B_0, double &B_1, double &B_2,
                          double &particle_E_coefficient)
      : particle_group(particle_group),
        sycl_target(particle_group->sycl_target), dt(dt), B_0(B_0), B_1(B_1),
        B_2(B_2), particle_E_coefficient(particle_E_coefficient) {}

  /**
   * Boris - First step.
   */
  inline void boris_1() {
    // A more advanced boris method may well have implementation here.
  }

  /**
   * Boris - Second step.
   */
  inline void boris_2() {
    auto t0 = NP::profile_timestamp();
    const double k_dt = this->dt;
    const double k_dht = this->dt * 0.5;
    const double k_B_0 = this->B_0;
    const double k_B_1 = this->B_1;
    const double k_B_2 = this->B_2;
    const NP::REAL k_E_coefficient = this->particle_E_coefficient;

    NP::particle_loop(
        "IntegratorBorisUniformB::boris_2", this->particle_group,
        [=](auto k_P, auto k_V, auto k_M, auto k_E, auto k_Q) {
          const NP::REAL Q = k_Q.at(0);
          const NP::REAL M = k_M.at(0);
          const NP::REAL QoM = Q / M;

          const NP::REAL scaling_t = QoM * k_dht;
          const NP::REAL t_0 = k_B_0 * scaling_t;
          const NP::REAL t_1 = k_B_1 * scaling_t;
          const NP::REAL t_2 = k_B_2 * scaling_t;

          const NP::REAL tmagsq = t_0 * t_0 + t_1 * t_1 + t_2 * t_2;
          const NP::REAL scaling_s = 2.0 / (1.0 + tmagsq);

          const NP::REAL s_0 = scaling_s * t_0;
          const NP::REAL s_1 = scaling_s * t_1;
          const NP::REAL s_2 = scaling_s * t_2;

          const NP::REAL V_0 = k_V.at(0);
          const NP::REAL V_1 = k_V.at(1);
          const NP::REAL V_2 = k_V.at(2);

          // The E dat contains d(phi)/dx not E -> multiply by -1.
          const NP::REAL v_minus_0 =
              V_0 + (-1.0 * k_E.at(0)) * scaling_t * k_E_coefficient;
          const NP::REAL v_minus_1 =
              V_1 + (-1.0 * k_E.at(1)) * scaling_t * k_E_coefficient;
          // E is zero in the z direction
          const NP::REAL v_minus_2 = V_2;

          NP::REAL v_prime_0, v_prime_1, v_prime_2;
          ELEC_PIC_2D3V_CROSS_PRODUCT_3D(v_minus_0, v_minus_1, v_minus_2, t_0,
                                         t_1, t_2, v_prime_0, v_prime_1,
                                         v_prime_2)

          v_prime_0 += v_minus_0;
          v_prime_1 += v_minus_1;
          v_prime_2 += v_minus_2;

          NP::REAL v_plus_0, v_plus_1, v_plus_2;
          ELEC_PIC_2D3V_CROSS_PRODUCT_3D(v_prime_0, v_prime_1, v_prime_2, s_0,
                                         s_1, s_2, v_plus_0, v_plus_1, v_plus_2)

          v_plus_0 += v_minus_0;
          v_plus_1 += v_minus_1;
          v_plus_2 += v_minus_2;

          // The E dat contains d(phi)/dx not E -> multiply by -1.
          k_V.at(0) =
              v_plus_0 + scaling_t * (-1.0 * k_E.at(0)) * k_E_coefficient;
          k_V.at(1) =
              v_plus_1 + scaling_t * (-1.0 * k_E.at(1)) * k_E_coefficient;
          // E is zero in the z direction
          k_V.at(2) = v_plus_2;

          // update of position to next time step
          k_P.at(0) += k_dt * k_V.at(0);
          k_P.at(1) += k_dt * k_V.at(1);
        },
        NP::Access::write(NP::Sym<NP::REAL>("P")),
        NP::Access::write(NP::Sym<NP::REAL>("V")),
        NP::Access::read(NP::Sym<NP::REAL>("M")),
        NP::Access::read(NP::Sym<NP::REAL>("E")),
        NP::Access::read(NP::Sym<NP::REAL>("Q")))
        ->execute();

    this->sycl_target->profile_map.inc(
        "IntegratorBorisUniformB", "Boris_2_Execute", 1,
        NP::profile_elapsed(t0, NP::profile_timestamp()));
  }
};

} // namespace NESO::Solvers::Electrostatic2D3V

#endif // __NESOSOLVERS_ELECTROSTATIC2D3V_BORISINTEGRATOR_HPP__
