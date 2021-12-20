class Mesh;

#ifndef __MESH_H__
#define __MESH_H__

#include <vector>
#include "plasma.hpp"
#include "fft.hpp"

class Mesh {
public:
	Mesh(int nintervals = 10);
	// time
	double t;
	// time step
	double dt;
	// number of time steps
	int nt;
	// number of grid spaces
    	int nintervals;
	// number of grid points
	int nmesh;
	// grid spacing
	double dx;
	// mesh point vector
	std::vector<double> mesh;
	// mesh point vector staggered at half points
	std::vector<double> mesh_staggered;
	// Fourier wavenumbers corresponding to mesh
	double *k;
	// Factor to use in the field solve
	double *poisson_factor;
	// Factor to use in combined field solve and E = -Grad(phi)
	double *poisson_E_factor;

	// charge density
	double *charge_density;
	// electric field
	double *electric_field;
	// electric field on a staggered grid
	double *electric_field_staggered;
	// electrostatic potential
	double *potential;

	// Calculate a particle's contribution to the electric field
	double evaluate_electric_field(const double x);

	// Deposit particle onto mesh
	void deposit(Plasma *plasma);

	// Solve the Gauss' law using finite differences
	void solve_for_potential();
	// Solve the Gauss' law using an FFT
	void solve_for_potential_fft(FFT *fft);
	// Solve the Gauss' law using an FFT and find E = - Grad(phi)
	void solve_for_electric_field_fft(FFT *fft);

	// Get electric field from the electrostatic potential
	void get_electric_field();
	// Interpolate E from unstaggered to staggered mesh
	void get_E_staggered_from_E();

	// Working arrays for the solver
	double *du, *d, *dl, *b;

	// Given a point x and a grid, find the indices of the grid points
	// either side of x
	int get_left_index(const double x, const std::vector<double> mesh);
};

#endif // __MESH_H__
