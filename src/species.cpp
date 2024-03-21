/*
 * Module for dealing with particles
 */

#include "species.hpp"
#include "mesh.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

/*
 * Initialize particles
 */
Species::Species(const Mesh &mesh, bool kinetic_in, double T_in, double q_in,
                 double m_in, int n_in)
    : kinetic(kinetic_in), T(T_in), q(q_in), m(m_in), vth(std::sqrt(2 * T / m)),
      n(n_in) {

  if (kinetic) {
    set_array_dimensions();
    set_initial_conditions(x, v);

    for (int i = 0; i < n; i++) {
      w[i] = 1.0 / double(n);
    }
  }
  // adiabatic species
  else {
    charge_density = q;
  }

  // Need to remove const if using adaptive timestepping
  dx_coef = mesh.dt * vth;
  dv_coef = 0.5 * mesh.dt * q / (m * vth);
}

/*
 * Set array dimensions for all the properties relating to the particles
 */
void Species::set_array_dimensions() {

  x.resize(n);   // particle x positions
  v.x.resize(n); // particle x velocities
  v.y.resize(n); // particle y velocities
  v.z.resize(n); // particle z velocities
  w.resize(n);   // particle weight
}

/*
 * Initialize distribution function
 * Pick random triplet (pos, vel, r) and keep particle if r < f(x,v)
 * for f the initial distribution.
 */
void Species::set_initial_conditions(std::vector<double> &x, Velocity &v) {

  // trial particle positions and velocities
  double pos, vel, r;
  // amplitude of wave perturbation
  // double amp = 1e-8;
  double big = 1e8;

  int i = 0;
  std::default_random_engine generator;
  while (i < n) {
    pos = std::uniform_real_distribution<double>(0.0, 1)(generator);
    vel = std::uniform_real_distribution<double>(-6.0, 6.0)(generator);
    r = std::uniform_real_distribution<double>(0.0, 1.0)(generator);

    // if( r * (1.0 + amp) < (1.0 + amp * cos( 2.0*M_PI*pos)) * exp(-vel*vel) /
    // sqrt(M_PI) ){ if( r * (1.0 + amp) < (1.0 + amp * cos( 2.0*M_PI*pos)) *
    // exp(-vel*vel) ){ if( r  <  amp * cos( 2.0*M_PI*pos) * exp(-vel*vel) /
    // sqrt(M_PI) ) { if( r < exp(-vel*vel) ) { if( r < 0.5 * big * ( exp(-
    // big*(vel-0.5)*(vel-0.5)) + exp(- big*(vel+0.5)*(vel+0.5)) )) {
    if (r < 0.5 * big *
                (exp(-big * (vel - 1.0) * (vel - 1.0)) +
                 exp(-big * (vel + 1.0) * (vel + 1.0)))) {
      x.at(i) = pos;
      v.x.at(i) = vel;
      v.y.at(i) = 0.0;
      v.z.at(i) = 0.0;
      i++;
    }
  }
}

/*
 * Second order accurate particle pusher
 * with spatially periodic boundary conditions
 */
void Species::push(sycl::queue &queue, Mesh *mesh) {
  sycl::buffer<double, 1> mesh_d(mesh->mesh.data(),
                                 sycl::range<1>{mesh->mesh.size()});
  sycl::buffer<double, 1> electric_field_d(
      mesh->electric_field.data(), sycl::range<1>{mesh->electric_field.size()});
  sycl::buffer<double, 1> x_d(x.data(), sycl::range<1>{x.size()});
  sycl::buffer<double, 1> vx_d(v.x.data(), sycl::range<1>{v.x.size()});

  const auto k_dx_coef = dx_coef;
  const auto k_dv_coef = dv_coef;
  const auto k_mesh_size = mesh->mesh.size();

  queue
      .submit([&](sycl::handler &cgh) {
        auto vx_a = vx_d.get_access<sycl::access::mode::read_write>(cgh);
        auto x_a = x_d.get_access<sycl::access::mode::read_write>(cgh);
        auto electric_field_a =
            electric_field_d.get_access<sycl::access::mode::read_write>(cgh);
        auto mesh_a = mesh_d.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::range{size_t(n)}, [=](sycl::id<1> idx) {
          const double F0 = Mesh1D::sycl_evaluate_electric_field(
              mesh_a, k_mesh_size, electric_field_a, x_a[idx]);
          // First half-push v
          vx_a[idx] += k_dv_coef * F0;

          // Push x
          x_a[idx] += k_dx_coef * vx_a[idx];
          while (x_a[idx] < 0) {
            x_a[idx] += 1.0;
          }
          x_a[idx] = std::fmod(x_a[idx], 1.0);

          const double F1 = Mesh1D::sycl_evaluate_electric_field(
              mesh_a, k_mesh_size, electric_field_a, x_a[idx]);
          // Second half-push v
          vx_a[idx] += k_dv_coef * F1;
        });
      })
      .wait();
}
