class Mesh;

#ifndef __MESH_H__
#define __MESH_H__

#include "plasma.hpp"

class Mesh {
public:
	Mesh();
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

	// charge density
	double *charge_density;
	// electric field
	double *electric_field;

	// Calculate a particle's contribution to the electric field
	double evaluate_electric_field(double *x);

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
