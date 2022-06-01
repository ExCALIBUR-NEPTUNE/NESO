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
  sycl::buffer<int, 1> nmesh_d;
  // grid spacing
  double dx;
  sycl::buffer<double, 1> dx_d;

  // box length in units of Debye length
  double normalized_box_length;
  // mesh point vector
  std::vector<double> mesh;
  // mesh points (device buffer)
  sycl::buffer<double, 1> mesh_d;
  // mesh point vector staggered at half points
  std::vector<double> mesh_staggered;
  // Fourier wavenumbers corresponding to mesh
  std::vector<double> k;
  // Factor to use in the field solve
  std::vector<double> poisson_factor;
  // Factor to use in combined field solve and E = -Grad(phi)
  std::vector<Complex> poisson_E_factor;
  sycl::buffer<Complex, 1> poisson_E_factor_d;

  // charge density
  std::vector<double> charge_density;
  sycl::buffer<double, 1> charge_density_d;
  // electric field
  std::vector<double> electric_field;
  // electric field (device)
  sycl::buffer<double, 1> electric_field_d;
  // electric field on a staggered grid
  std::vector<double> electric_field_staggered;
  // electrostatic potential
  std::vector<double> potential;

  // Calculate a particle's contribution to the electric field
  double evaluate_electric_field(const double x);

#ifdef NESO_DPCPP
  SYCL_EXTERNAL double
  sycl_evaluate_electric_field(sycl::accessor<double> mesh_d,
                               sycl::accessor<double> electric_field_d,
                               double x);
#else
  double sycl_evaluate_electric_field(sycl::accessor<double> mesh_d,
                                      sycl::accessor<double> electric_field_d,
                                      double x);
#endif


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

#ifdef NESO_DPCPP
  SYCL_EXTERNAL int sycl_get_left_index(const double x,
                                        const sycl::accessor<double> mesh_d);
#else
  int sycl_get_left_index(const double x,
                          const sycl::accessor<double> mesh_d);
#endif

};

#endif // __MESH_H__
