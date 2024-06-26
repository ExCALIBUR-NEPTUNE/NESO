#ifndef __SPECIES_H__
#define __SPECIES_H__
class Species;

#include "mesh.hpp"
#include "sycl_typedefs.hpp"
#include "velocity.hpp"

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
  // particle velocity structure of arrays
  Velocity v;
  // charge density of species (if adiabatic)
  double charge_density;
  // particle position array at
  // next timestep
  std::vector<double> xnew;
  // particle velocity array at
  // next tmiestep
  std::vector<double> vnew;
  // particle weight
  std::vector<double> w;
  // particle pusher
  void push(sycl::queue &q, Mesh *mesh);
  // set array sizes for particle properties
  void set_array_dimensions();
  // initial conditions
  void set_initial_conditions(std::vector<double> &x, Velocity &v);
  // Coefficients for particle pusher
  double dx_coef;
  double dv_coef;
};

#endif // __SPECIES_H__
