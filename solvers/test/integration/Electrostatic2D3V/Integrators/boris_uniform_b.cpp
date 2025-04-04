#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <random>

#include "../../../../../solvers/Electrostatic2D3V/ParticleSystems/IntegratorBorisUniformB.hpp"

using namespace NESO::Particles;
namespace ES2D3V = NESO::Solvers::Electrostatic2D3V;

inline double get_B_error(const int N, const int N_step, double dt) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 8;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 0;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto domain = std::make_shared<Domain>(mesh);

  ParticleSpec particle_spec{
      NP::ParticleProp(NP::Sym<NP::REAL>("P"), ndim, true),
      NP::ParticleProp(NP::Sym<NP::REAL>("P_CORRECT"), 2),
      NP::ParticleProp(NP::Sym<NP::REAL>("V"), 3),
      NP::ParticleProp(NP::Sym<NP::INT>("CELL_ID"), 1, true),
      NP::ParticleProp(NP::Sym<NP::REAL>("M"), 1),
      NP::ParticleProp(NP::Sym<NP::REAL>("E"), ndim),
      NP::ParticleProp(NP::Sym<NP::REAL>("Q"), 1),
      NP::ParticleProp(NP::Sym<NP::REAL>("P_ORIG"), ndim)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  if (sycl_target->comm_pair.rank_parent == 0) {
    std::mt19937 rng_pos(52234234);

    double extents[2] = {2.0, 2.0};
    auto positions = NP::uniform_within_extents(N, ndim, extents, rng_pos);
    NP::ParticleSet initial_distribution(N, A->get_particle_spec());

    for (int px = 0; px < N; px++) {
      const double x = positions[0][px] + 2.0;
      const double y = positions[1][px] + 2.0;

      initial_distribution[NP::Sym<NP::REAL>("P")][px][0] = x;
      initial_distribution[NP::Sym<NP::REAL>("P_ORIG")][px][0] = x;
      initial_distribution[NP::Sym<NP::REAL>("P")][px][1] = y;
      initial_distribution[NP::Sym<NP::REAL>("P_ORIG")][px][1] = y;
      initial_distribution[NP::Sym<NP::REAL>("V")][px][0] = 1.0;
      initial_distribution[NP::Sym<NP::REAL>("V")][px][1] = 0.0;
      initial_distribution[NP::Sym<NP::REAL>("V")][px][2] = 0.0;
      initial_distribution[NP::Sym<NP::REAL>("M")][px][0] = 1.0;
      initial_distribution[NP::Sym<NP::REAL>("Q")][px][0] = 1.0;
      initial_distribution[NP::Sym<NP::INT>("CELL_ID")][px][0] = 0;
    }

    A->add_particles_local(initial_distribution);
  }

  A->global_move();

  double B_0 = 0.0;
  double B_1 = 0.0;
  double B_2 = 1.0;
  double particle_E_coefficient = 0.0;

  auto integrator_boris = std::make_shared<ES2D3V::IntegratorBorisUniformB>(
      A, dt, B_0, B_1, B_2, particle_E_coefficient);

  double T = 0.0;
  const int cell_count = domain->mesh->get_cell_count();
  double last_mean_err;
  auto test_positions = [&]() -> void {
    double mean_error = 0.0;
    int mean_count = 0;

    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto P = (*A)[NP::Sym<NP::REAL>("P")]->cell_dat.get_cell(cellx);
      auto P_ORIG = (*A)[NP::Sym<NP::REAL>("P_ORIG")]->cell_dat.get_cell(cellx);
      auto P_CORRECT =
          (*A)[NP::Sym<NP::REAL>("P_CORRECT")]->cell_dat.get_cell(cellx);
      const int nrow = P->nrow;
      const double radius = 1.0;
      const double pi = 3.14159265359;
      const double two_pi = 2.0 * pi;
      const double angular_velocity = (1.0 / (two_pi * radius)) * two_pi;

      const double extent0 = mesh->global_extents[0];
      const double extent1 = mesh->global_extents[1];

      // for each particle
      for (int px = 0; px < nrow; px++) {
        double origin[2];
        origin[0] = (*P_ORIG)[0][px];
        origin[1] = (*P_ORIG)[1][px] - 1.0;

        const double correct_angle = angular_velocity * T;

        const double correct_x = std::fmod(
            origin[0] + std::sin(correct_angle) * radius + extent0, extent0);
        const double correct_y = std::fmod(
            origin[1] + std::cos(correct_angle) * radius + extent1, extent1);

        const double x = (*P)[0][px];
        const double y = (*P)[1][px];

        const double err_x = ABS(x - correct_x);
        const double err_y = ABS(y - correct_y);

        (*P_CORRECT)[0][px] = correct_x;
        (*P_CORRECT)[1][px] = correct_y;

        mean_error += std::sqrt(err_x * err_x + err_y * err_y);
        mean_count += 1;

        ASSERT_TRUE(err_x < 2.0e-3);
        ASSERT_TRUE(err_y < 2.0e-3);
      }

      (*A)[NP::Sym<NP::REAL>("P_CORRECT")]->cell_dat.set_cell(cellx, P_CORRECT);
    }

    double global_mean_error;
    int global_mean_count;

    MPICHK(MPI_Allreduce(&mean_error, &global_mean_error, 1, MPI_DOUBLE,
                         MPI_SUM, mesh->get_comm()));
    MPICHK(MPI_Allreduce(&mean_count, &global_mean_count, 1, MPI_INT, MPI_SUM,
                         mesh->get_comm()));

    last_mean_err = global_mean_error / global_mean_count;
  };

  for (int stepx = 0; stepx < N_step; stepx++) {
    T += dt;
    integrator_boris->boris_1();
    integrator_boris->boris_2();
    A->global_move();
    if (stepx % 40 == 0) {
      test_positions();
    }
  }

  test_positions();
  A->free();
  mesh->free();

  return last_mean_err;
}

TEST(Electrostatic2D3V, IntegratorBorisUniformB_1) {

  const int N = 10;

  const double err3 = get_B_error(N, 1000, 0.001);
  const double err32 = get_B_error(N, 1000 * 2, 0.001 / 2.0);

  // Test for first order accuracy
  ASSERT_NEAR(err3 / err32, 2.0, 1.0e-3);

  const double err38 = get_B_error(N, 1000 * 8, 0.001 / 8.0);

  // Test for first order accuracy
  ASSERT_NEAR(err3 / err38, 8.0, 1.0e-3);
}
