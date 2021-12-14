class Mesh;

#ifndef __MESH_H__
#define __MESH_H__

#include "plasma.hpp"

class Mesh {
public:
	Mesh(int ninterval = 10);
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
	double *mesh;
	// mesh point vector staggered at half points
	double *mesh_staggered;

	// charge density
	double *charge_density;
	// electric field
	double *electric_field;
	// electric field on a staggered grid
	double *electric_field_staggered;

	// Calculate a particle's contribution to the electric field
	double evaluate_electric_field(const double x);

	// Deposit particle onto mesh
	void deposit(Plasma *plasma);

	// Solve the Gauss' law
	void solve(Plasma *plasma);

	// Get electric field from the electrostatic potential
	void get_electric_field(double *potential);

	// Working arrays for the solver
	double *du, *d, *dl, *b;

};

#endif // __MESH_H__
