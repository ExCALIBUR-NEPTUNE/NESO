#ifndef __BORIS_INTEGRATOR_H_
#define __BORIS_INTEGRATOR_H_

#include <cassert>

#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

#ifndef PIC_2D3V_CROSS_PRODUCT_3D
#define PIC_2D3V_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)          \
  (c1) = ((a2) * (b3)) - ((a3) * (b2));                                        \
  (c2) = ((a3) * (b1)) - ((a1) * (b3));                                        \
  (c3) = ((a1) * (b2)) - ((a2) * (b1));
#endif

class IntegratorBoris {

private:
  ParticleGroupSharedPtr particle_group;
  SYCLTargetSharedPtr sycl_target;

  double dt;
  std::array<double, 3> B0;

public:
  IntegratorBoris(ParticleGroupSharedPtr particle_group, double &dt,
      const std::array<double, 3>& B0 )
      : particle_group(particle_group),
        sycl_target(particle_group->sycl_target), dt(dt), B0(B0) {}

  /**
   * Move particles according to their velocity a fraction of dt.
   *  @param dt_fraction The fraction of dt to advect particles by (default=1.0)
   */
  inline void advect(const double dt_fraction = 1.0) {
    auto t0 = profile_timestamp();

    auto k_X = (*this->particle_group)[Sym<REAL>("X")]->cell_dat.device_ptr();
    const auto k_V =
        (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_V_OLD =
        (*this->particle_group)[Sym<REAL>("V_OLD")]->cell_dat.device_ptr();
    const auto k_Q =
        (*this->particle_group)[Sym<INT>("Q")]->cell_dat.device_ptr();

    const auto k_W =
        (*this->particle_group)[Sym<REAL>("W")]->cell_dat.device_ptr();
    auto k_WQ = (*this->particle_group)[Sym<REAL>("WQ")]->cell_dat.device_ptr();
    auto k_WQV =
        (*this->particle_group)[Sym<REAL>("WQV")]->cell_dat.device_ptr();

    const double k_dt = this->dt * dt_fraction;

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    this->sycl_target->profile_map.inc(
        "IntegratorBoris", "Advection", 1,
        profile_elapsed(t0, profile_timestamp()));
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          // sycl::stream out(1024, 256, cgh);
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const REAL vx = k_V[cellx][0][layerx];
                const REAL vy = k_V[cellx][1][layerx];

                assert(std::isfinite(vx));
                assert(std::isfinite(vy));
                // update of position to next time step
                k_X[cellx][0][layerx] += k_dt * vx;
                k_X[cellx][1][layerx] += k_dt * vy;

                // update the charge and current particle dats
                const REAL wq = k_W[cellx][0][layerx] * k_Q[cellx][0][layerx];
                k_WQ[cellx][0][layerx] = wq;
                k_WQV[cellx][0][layerx] = wq * k_V[cellx][0][layerx];
                k_WQV[cellx][1][layerx] = wq * k_V[cellx][1][layerx];
                k_WQV[cellx][2][layerx] = wq * k_V[cellx][2][layerx];

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    this->sycl_target->profile_map.inc(
        "IntegratorBoris", "Boris_2_Execute", 1,
        profile_elapsed(t0, profile_timestamp()));
  }

  /**
   * Boris - mutate velocities only
   */
  inline void accelerate(const double dt_fraction) {
    auto t0 = profile_timestamp();

    //    auto k_X =
    //    (*this->particle_group)[Sym<REAL>("X")]->cell_dat.device_ptr();
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();

    const auto k_V_OLD =
        (*this->particle_group)[Sym<REAL>("V_OLD")]->cell_dat.device_ptr();

    const auto k_M =
        (*this->particle_group)[Sym<INT>("M")]->cell_dat.device_ptr();
    const auto k_GradAx =
        (*this->particle_group)[Sym<REAL>("GradAx")]->cell_dat.device_ptr();
    const auto k_GradAy =
        (*this->particle_group)[Sym<REAL>("GradAy")]->cell_dat.device_ptr();
    const auto k_GradAz =
        (*this->particle_group)[Sym<REAL>("GradAz")]->cell_dat.device_ptr();
    const auto k_B =
        (*this->particle_group)[Sym<REAL>("B")]->cell_dat.device_ptr();
    const auto k_E =
        (*this->particle_group)[Sym<REAL>("E")]->cell_dat.device_ptr();
    const auto k_Q =
        (*this->particle_group)[Sym<INT>("Q")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    const auto k_B0x = this->B0[0];
    const auto k_B0y = this->B0[1];
    const auto k_B0z = this->B0[2];

    const double k_dt = this->dt * dt_fraction;
    const double k_dht = k_dt * 0.5;

    this->sycl_target->profile_map.inc(
        "IntegratorBoris", "what this for?", 1,
        profile_elapsed(t0, profile_timestamp()));
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          // sycl::stream out(1024, 256, cgh);
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const INT Q = k_Q[cellx][0][layerx];
                const INT M = k_M[cellx][0][layerx];
                const REAL QoM = REAL(Q) / REAL(M);

                const REAL scaling_t = QoM * k_dht;
                const REAL Bx = k_GradAz[cellx][1][layerx] + k_B0x;
                const REAL By = - k_GradAz[cellx][0][layerx] + k_B0y;
                const REAL Bz = k_GradAy[cellx][0][layerx] - k_GradAx[cellx][1][layerx] + k_B0z;

                // save for diagnostics, but not strictly required to save on particle
                k_B[cellx][0][layerx] = Bx;
                k_B[cellx][1][layerx] = By;
                k_B[cellx][2][layerx] = Bz;

                const REAL t_0 = Bx * scaling_t;
                const REAL t_1 = By * scaling_t;
                const REAL t_2 = Bz * scaling_t;

                const REAL tmagsq = t_0 * t_0 + t_1 * t_1 + t_2 * t_2;
                const REAL scaling_s = 2.0 / (1.0 + tmagsq);

                const REAL s_0 = scaling_s * t_0;
                const REAL s_1 = scaling_s * t_1;
                const REAL s_2 = scaling_s * t_2;

                REAL V_0 = k_V[cellx][0][layerx];
                REAL V_1 = k_V[cellx][1][layerx];
                REAL V_2 = k_V[cellx][2][layerx];

                k_V_OLD[cellx][0][layerx] = V_0;
                k_V_OLD[cellx][1][layerx] = V_1;
                k_V_OLD[cellx][2][layerx] = V_2;

                REAL gamma = 1 / sqrt(1 - V_0 * V_0 - V_1 * V_1 - V_2 * V_2);
                assert(std::isfinite(gamma));

                const REAL Ex = k_E[cellx][0][layerx];
                const REAL Ey = k_E[cellx][1][layerx];
                const REAL Ez = k_E[cellx][2][layerx];

                // Strictly speaking, v_minus_*, v_prime_*, v_plus_* are velocities
                // multiplied by relativistic gamma
                const REAL v_minus_0 = V_0 * gamma + Ex * scaling_t;
                const REAL v_minus_1 = V_1 * gamma + Ey * scaling_t;
                const REAL v_minus_2 = V_2 * gamma + Ez * scaling_t;

                REAL v_prime_0, v_prime_1, v_prime_2;
                PIC_2D3V_CROSS_PRODUCT_3D(v_minus_0, v_minus_1, v_minus_2, t_0,
                                          t_1, t_2, v_prime_0, v_prime_1,
                                          v_prime_2)

                v_prime_0 += v_minus_0;
                v_prime_1 += v_minus_1;
                v_prime_2 += v_minus_2;

                REAL v_plus_0, v_plus_1, v_plus_2;
                PIC_2D3V_CROSS_PRODUCT_3D(v_prime_0, v_prime_1, v_prime_2, s_0,
                                          s_1, s_2, v_plus_0, v_plus_1,
                                          v_plus_2)

                v_plus_0 += v_minus_0;
                v_plus_1 += v_minus_1;
                v_plus_2 += v_minus_2;

                V_0 = v_plus_0 + scaling_t * Ex;
                V_1 = v_plus_1 + scaling_t * Ey;
                V_2 = v_plus_2 + scaling_t * Ez;

                gamma = sqrt(1 + V_0 * V_0 + V_1 * V_1 + V_2 * V_2);

                assert(std::isfinite(V_0));
                assert(std::isfinite(V_1));
                assert(std::isfinite(V_2));

                k_V[cellx][0][layerx] = V_0 / gamma;
                k_V[cellx][1][layerx] = V_1 / gamma;
                k_V[cellx][2][layerx] = V_2 / gamma;

                //                out << "Ex = " << k_E[cellx][0][layerx] <<
                //                       "Ey = " << k_E[cellx][1][layerx] <<
                //                       "Ez = " << k_E[cellx][2][layerx] <<
                //                       cl::sycl::endl;

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    this->sycl_target->profile_map.inc(
        "IntegratorBoris", "Boris_2_Execute", 1,
        profile_elapsed(t0, profile_timestamp()));
  }
};

#endif
