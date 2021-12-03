/*
 * Module for dealing with particles
 */

#include "mesh.hpp"
#include "plasma.hpp"
#include <string>
#include <iostream>
#include <cmath>

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
	charge_density = new double[nmesh];
	electric_field = new double[nmesh];
	for( int i = 0; i < nmesh; i++){
        	charge_density[i] = 0.0;
        	electric_field[i] = 0.0;
	}

	// super diagonal
	du = new double [nmesh-1];
	// sub diagonal
	dl = new double [nmesh-1];
	// diagonal
	d = new double [nmesh];
	// right hand side: - dx^2 * charge density
	b = new double [nmesh];
}

// Invert real double tridiagonal matrix with lapack
extern "C" {
  	//int dgttrs_(char trans, int n, int nrhs, const double *dl, const double *d, const double *du, const double *du2, const int *ipiv, double *b, int ldb);
        void dgtsv_(int *n, int *nrhs, double *dl, double *d, double *du, double *b, int *ldb, int *info); 
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

	for(int i = 0; i < plasma->n; i++) {
		// get index of left-hand grid point
		//std::cout << plasma->x[i] << "\n";
		int index = (floor(plasma->x[i]/dx));
		//std::cout << index << "\n";
		// r is the proportion if the distance into the cell that the particle is at
		// e.g. midpoint => r = 0.5
		double r = plasma->x[i] / dx - double(index);
		//std::cout << r << "\n\n";
		charge_density[index] += (1.0-r) * plasma->w[i] / dx;
		charge_density[index+1] += r * plasma->w[i] / dx;
	}

	//for( int i = 0; i < nmesh; i++){
	//	std::cout << charge_density[i] << "\n";
	//}
}

/*
 * Solve Gauss' law for the electrostatic potential using the charge
 * distribution as a solve. In 1D, this is a tridiagonal matrix inversion with
 * the Thomas algorithm.
 */
void Mesh::solve(Plasma *plasma) {

	// Initialize with general terms
	for(int i = 0; i < nmesh - 1; i++) {
        	du[i] = 1.0;
        	dl[i] = 1.0;
        	d[i] = -2.0;
        	b[i] = - dx * dx * charge_density[i];
	}
        d[nmesh-1] = -2.0;
        b[nmesh-1] = - dx * dx * charge_density[nmesh-1];

	// apply boundary conditions
	// for simplicity use zero boundary conditions -
	// this is inconsistent with the periodic
	// particle pusher but allows us to use simple
	// tridiagonal inversion
	d[0] = 1.0;
	du[0] = 0.0;
	b[0] = 0.0;

	d[nmesh-1] = 1.0;
	dl[nmesh-2] = 0.0; // highest element, dl is offset by 1
	b[nmesh-1] = 0.0;

	int info;
	int nrhs = 1;
	int ldb = nmesh;
  	dgtsv_(&nmesh, &nrhs, dl, d, du, b, &ldb, &info);
        
        // compute gradient of potential on half-mesh
        get_electric_field(b);

}
        
/*
 * Find the electric field by taking the gradient of the potential
 */
void Mesh::get_electric_field(double *potential) {
	for( int i = 0; i < nmesh-1; i++){
        	electric_field[i] = -( potential[i+1] - potential[i] ) / dx;
	}
}
