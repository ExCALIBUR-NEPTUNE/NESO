class Species;

#ifndef __SPECIES_H__
#define __SPECIES_H__

#include "mesh.hpp"
#include "velocity.hpp"

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

class Species {
public:
  Species(const Mesh &mesh, bool kinetic = true, double T = 1.0, double q = 1,
          double m = 1, int n = 10);
  // Whether this species is treated kinetically (true) or adiabatically (false)
  bool kinetic;
  // number of particles
  int n;
  // temperature
  double T;
  // charge
  int q;
  // mass
  double m;
  // thermal velocity
  double vth;
  // particle position array
  std::vector<double> x;
  // particle positions (device)
  sycl::buffer<double, 1> x_d;
  // particle velocity structure of arrays
  Velocity v;
  // particle x velocity (device)
  sycl::buffer<double, 1> vx_d;
  // particle y velocity (device)
  sycl::buffer<double, 1> vy_d;
  // particle z velocity (device)
  sycl::buffer<double, 1> vz_d;
  // charge density of species (if adiabatic)
  double charge_density;
  sycl::buffer<double, 1> charge_density_d;
  // particle position array at
  // next timestep
  std::vector<double> xnew;
  // particle velocity array at
  // next tmiestep
  std::vector<double> vnew;
  // particle weight
  std::vector<double> w;
  // particle weights (device)
  sycl::buffer<double, 1> w_d;
  // particle pusher
  void push(sycl::queue &q, Mesh *mesh);
  // set array sizes for particle properties
  void set_array_dimensions();
  // initial conditions
  void set_initial_conditions(std::vector<double> &x, Velocity &v);
  // Coefficients for particle pusher
  double dx_coef;
  double dv_coef;
  sycl::buffer<double, 1> dx_coef_d;
  sycl::buffer<double, 1> dv_coef_d;
};

#endif // __SPECIES_H__
