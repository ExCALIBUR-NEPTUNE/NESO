/*
 * Module for dealing with particles
 */

#include "mesh.hpp"
#include "plasma.hpp"
#include <string>
#include <iostream>

/*
 * Initialize mesh
 */
Mesh::Mesh() {
	
  	// time
        t = 0.0;
	// time step
        dt = 0.1;
  	// number of time steps
        nt = 10;
	// number of grid spaces
	nintervals = 10;
	// number of grid points
        nmesh = nintervals + 1;
	// size of grid spaces on a domain of length 1
        dx = 1.0 / double(nintervals);

	// electric field on mesh points
	electric_field = new double[nmesh];
	for( int i = 0; i < nmesh; i++){
        	electric_field[i] = 0.0;
	}
}

/* 
 * Evaluate the electric field at x grid points by interpolating using the
 * values at staggered grid points
 */
double Mesh::evaluate_electric_field(double *x){
	std::cout<<"TODO: Implement interpolation of electric field.\n";

	return 1;
};

/*
 * Deposit charge at grid points. For a particle in a grid space [a,b],
 * distribute charge at grid points a and b proportionally to the particle's
 * distance from those points.
 */
void Mesh::deposit(Plasma *plasma){
	std::cout<<"TODO Implement deposit\n";
}

/*
 * Solve Gauss' law for the electrostatic potential using the charge
 * distribution as a solve. In 1D, this is a tridiagonal matrix inversion with
 * the Thomas algorithm.
 */
void Mesh::solve(Plasma *plasma) {
	std::cout<<"TODO Implement solver\n";

	// At end of this routine, we have the electrostatic potential. 
	// Call get_electric_field to calculate the electric field from the
	// potential
}
        
/*
 * Find the electric field by taking the gradient of the potential
 */
void Mesh::get_electric_field() {
	std::cout << "TODO: Implement get_electic_field\n";
}
