class Mesh;

#ifndef __MESH_H__
#define __MESH_H__

#include "plasma.hpp"
#include <vector>

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

	// Solve the Gauss' law
	void solve_for_potential();

	// Get electric field from the electrostatic potential
	void get_electric_field();

	// Working arrays for the solver
	double *du, *d, *dl, *b;

	// Given a point x and a grid, find the indices of the grid points
	// either side of x
	void get_index_pair(const double x, const std::vector<double> mesh, const int meshsize, int *index_down, int *index_up);
};

#endif // __MESH_H__
