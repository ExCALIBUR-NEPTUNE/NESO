class Mesh;

#ifndef __MESH_H__
#define __MESH_H__

#include "custom_types.hpp"
#include "fft_wrappers.hpp"
#include "plasma.hpp"
#include "species.hpp"
#include <vector>

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

class Mesh {
public:
  Mesh(int nintervals = 10, double dt = 0.1, int nt = 1000);
  // time
  double t;
  // time step
  double dt;
  // number of time steps
  int nt;
  // number of grid spaces
  int nintervals;
  // number of grid points (including periodic point)
  int nmesh;
  // grid spacing
  double dx;

  // box length in units of Debye length
  double normalized_box_length;
  // mesh point vector
  std::vector<double> mesh;
  // mesh point vector staggered at half points
  std::vector<double> mesh_staggered;
  // Fourier wavenumbers corresponding to mesh
  std::vector<double> k;
  // Factor to use in the field solve
  std::vector<double> poisson_factor;
  // Factor to use in combined field solve and E = -Grad(phi)
  std::vector<Complex> poisson_E_factor;

  // charge density
  std::vector<double> charge_density;
  // electric field
  std::vector<double> electric_field;
  // electric field on a staggered grid
  std::vector<double> electric_field_staggered;
  // electrostatic potential
  std::vector<double> potential;

  // Calculate a particle's contribution to the electric field
  double evaluate_electric_field(const double x);

  // Deposit particle onto mesh
  void deposit(Plasma &plasma);
  void sycl_deposit(sycl::queue &Q, Plasma &plasma);

  // Solve the Gauss' law using finite differences
  void solve_for_potential();
  // Solve the Gauss' law using an FFT
  // void solve_for_potential_fft(FFT &fft);
  // Solve the Gauss' law using an FFT and find E = - Grad(phi)
  // void solve_for_electric_field_fft(FFT &fft);
  void sycl_solve_for_electric_field_fft(sycl::queue &Q, FFT &fft);

  // Get electric field from the electrostatic potential
  void get_electric_field();
  // Interpolate E from unstaggered to staggered mesh
  void get_E_staggered_from_E();

  // Set the electric field consistently with the particles
  void set_initial_field(sycl::queue &Q, Mesh &mesh, Plasma &plasma, FFT &fft);

  // Working arrays for the solver
  // NB must be double * for use in lapack call
  double *du, *d, *dl, *b;

  // Given a point x and a grid, find the indices of the grid points
  // either side of x
  int get_left_index(const double x, const std::vector<double> mesh);
};

namespace Mesh1D {

template <typename T>
inline int sycl_get_left_index(const double x, const T &mesh_d,
                               const int mesh_size) {

  int index = 0;
  while ((mesh_d[index + 1] < x) and (index < mesh_size)) {
    index++;
  };
  return index;
}

/*
 * Evaluate the electric field at x grid points by
 * interpolating onto the grid
 * SYCL note: this is a copy of evaluate_electric_field, but able to be called
 * in sycl. This should become evaluate_electric_field eventually
 */
inline double sycl_evaluate_electric_field(
    const sycl::accessor<double> &mesh_d, const int mesh_size,
    const sycl::accessor<double> &electric_field_d, double x) {

  // Find grid cell that x is in
  int index = sycl_get_left_index(x, mesh_d, mesh_size);

  // now x is in the cell ( mesh[index-1], mesh[index] )

  double cell_width = mesh_d[index + 1] - mesh_d[index];
  double distance_into_cell = x - mesh_d[index];

  // r is the proportion if the distance into the cell that the particle is at
  // e.g. midpoint => r = 0.5
  double r = distance_into_cell / cell_width;

  return (1.0 - r) * electric_field_d[index] + r * electric_field_d[index + 1];
}

} // namespace Mesh1D

#endif // __MESH_H__
