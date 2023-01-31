#ifndef __BORIS_INTEGRATOR_H_
#define __BORIS_INTEGRATOR_H_

#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

#ifndef ELEC_PIC_2D3V_CROSS_PRODUCT_3D
#define ELEC_PIC_2D3V_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)     \
  (c1) = ((a2) * (b3)) - ((a3) * (b2));                                        \
  (c2) = ((a3) * (b1)) - ((a1) * (b3));                                        \
  (c3) = ((a1) * (b2)) - ((a2) * (b1));
#endif

class IntegratorBorisUniformB {

private:
  ParticleGroupSharedPtr particle_group;
  SYCLTargetSharedPtr sycl_target;

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
  inline void set_B_field(const REAL B0 = 0.0, const REAL B1 = 0.0,
                          const REAL B2 = 0.0) {
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
  inline void set_E_coefficent(const REAL x) {
    this->particle_E_coefficient = x;
  }

  IntegratorBorisUniformB(ParticleGroupSharedPtr particle_group, double &dt,
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
    auto t0 = profile_timestamp();

    auto k_P = (*this->particle_group)[Sym<REAL>("P")]->cell_dat.device_ptr();
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_M =
        (*this->particle_group)[Sym<REAL>("M")]->cell_dat.device_ptr();
    const auto k_E =
        (*this->particle_group)[Sym<REAL>("E")]->cell_dat.device_ptr();
    const auto k_Q =
        (*this->particle_group)[Sym<REAL>("Q")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    const double k_dt = this->dt;
    const double k_dht = this->dt * 0.5;
    const double k_B_0 = this->B_0;
    const double k_B_1 = this->B_1;
    const double k_B_2 = this->B_2;
    const REAL k_E_coefficient = this->particle_E_coefficient;

    this->sycl_target->profile_map.inc(
        "IntegratorBorisUniformB", "VelocityVerlet_2_Prepare", 1,
        profile_elapsed(t0, profile_timestamp()));
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const REAL Q = k_Q[cellx][0][layerx];
                const REAL M = k_M[cellx][0][layerx];
                const REAL QoM = Q / M;

                const REAL scaling_t = QoM * k_dht;
                const REAL t_0 = k_B_0 * scaling_t;
                const REAL t_1 = k_B_1 * scaling_t;
                const REAL t_2 = k_B_2 * scaling_t;

                const REAL tmagsq = t_0 * t_0 + t_1 * t_1 + t_2 * t_2;
                const REAL scaling_s = 2.0 / (1.0 + tmagsq);

                const REAL s_0 = scaling_s * t_0;
                const REAL s_1 = scaling_s * t_1;
                const REAL s_2 = scaling_s * t_2;

                const REAL V_0 = k_V[cellx][0][layerx];
                const REAL V_1 = k_V[cellx][1][layerx];
                const REAL V_2 = k_V[cellx][2][layerx];

                // The E dat contains d(phi)/dx not E -> multiply by -1.
                const REAL v_minus_0 = V_0 + (-1.0 * k_E[cellx][0][layerx]) *
                                                 scaling_t * k_E_coefficient;
                const REAL v_minus_1 = V_1 + (-1.0 * k_E[cellx][1][layerx]) *
                                                 scaling_t * k_E_coefficient;
                // E is zero in the z direction
                const REAL v_minus_2 = V_2;

                REAL v_prime_0, v_prime_1, v_prime_2;
                ELEC_PIC_2D3V_CROSS_PRODUCT_3D(v_minus_0, v_minus_1, v_minus_2,
                                               t_0, t_1, t_2, v_prime_0,
                                               v_prime_1, v_prime_2)

                v_prime_0 += v_minus_0;
                v_prime_1 += v_minus_1;
                v_prime_2 += v_minus_2;

                REAL v_plus_0, v_plus_1, v_plus_2;
                ELEC_PIC_2D3V_CROSS_PRODUCT_3D(v_prime_0, v_prime_1, v_prime_2,
                                               s_0, s_1, s_2, v_plus_0,
                                               v_plus_1, v_plus_2)

                v_plus_0 += v_minus_0;
                v_plus_1 += v_minus_1;
                v_plus_2 += v_minus_2;

                // The E dat contains d(phi)/dx not E -> multiply by -1.
                k_V[cellx][0][layerx] =
                    v_plus_0 + scaling_t * (-1.0 * k_E[cellx][0][layerx]) *
                                   k_E_coefficient;
                k_V[cellx][1][layerx] =
                    v_plus_1 + scaling_t * (-1.0 * k_E[cellx][1][layerx]) *
                                   k_E_coefficient;
                // E is zero in the z direction
                k_V[cellx][2][layerx] = v_plus_2;

                // update of position to next time step
                k_P[cellx][0][layerx] += k_dt * k_V[cellx][0][layerx];
                k_P[cellx][1][layerx] += k_dt * k_V[cellx][1][layerx];

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    this->sycl_target->profile_map.inc(
        "IntegratorBorisUniformB", "Boris_2_Execute", 1,
        profile_elapsed(t0, profile_timestamp()));
  }
};

#endif
