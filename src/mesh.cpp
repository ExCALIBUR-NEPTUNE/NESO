/*
 * Module for dealing with particles
 */

#include "mesh.hpp"
#include "plasma.hpp"
#include "fft.hpp"
#include <string>
#include <iostream>
#include <cmath>
#include <fftw3.h>

/*
 * Initialize mesh
 */
Mesh::Mesh(int nintervals_in) {
	
  	// time
        t = 0.0;
	// time step
        dt = 0.01;
  	// number of time steps
        nt = 1000;
	// number of grid points
        nintervals = nintervals_in;
	// number of grid points (including periodic point)
        nmesh = nintervals + 1;
	// size of grid spaces on a domain of length 1
        dx = 1.0 / double(nintervals);
	
	// mesh point vector
	mesh = new double[nmesh];
	// mesh point vector staggered at half points
	mesh_staggered = new double[nmesh-1];
	for( int i = 0; i < nmesh; i++){
        	mesh[i] = double(i)*dx;
	}
	for( int i = 0; i < nmesh-1; i++){
        	mesh_staggered[i] = double(i+0.5)*dx;
	}
	// Fourier wavenumbers k[j] =
	// 2*pi*j such that k*mesh gives
	// the argument of the exponential
	// in FFTW
	// NB: for complex to complex transforms, the second half of Fourier
	// modes must be negative of the first half
	k = new double[nmesh];
	for( int i = 0; i < (nmesh/2)+1; i++){
        	k[i] = 2.0*M_PI*double(i);
        	k[nmesh-i-1] = -k[i];
	}
	// Poisson factor
	// Coefficient to multiply
	// Fourier-transformed charge
	// density by in the Poisson solve:
	// - 1 / (k**2 * nmesh)
	// This accounts for the change of
	// sign, the wavenumber squared
	// (for the Laplacian), and the
	// length of the array (the
	// normalization in the FFT).
	poisson_factor = new double[nintervals];
        poisson_factor[0] = 0.0;
	for( int i = 1; i < nintervals; i++){
        	poisson_factor[i] = -1.0/(k[i]*k[i]*double(nintervals));
	}

	// Poisson Electric field factor
	// Coefficient to multiply
	// Fourier-transformed charge
	// density by in the Poisson solve:
	// i / (k * nmesh)
	// to obtain the electric field from the Poisson equation and 
	// E = - Grad(phi) in a single step.
	// This accounts for the change of sign, the wavenumber squared (for
	// the Laplacian), the length of the array (the normalization in the
	// FFT), and the factor of (-ik) for taking the Grad in fourier space.
	// NB Rather than deal with complex arithmetic, we make this a real
	// number that we apply to the relevant array entry.
	poisson_E_factor = new double[nintervals];
        poisson_E_factor[0] = 0.0;
	for( int i = 1; i < nintervals; i++){
        	poisson_E_factor[i] = 1.0/(k[i]*double(nintervals));
	}

	charge_density = new double[nmesh];
	for( int i = 0; i < nmesh; i++){
        	charge_density[i] = 0.0;
	}
	// Electric field on mesh
	electric_field = new double[nmesh-1];
	for( int i = 0; i < nmesh; i++){
        	electric_field[i] = 0.0;
	}
	// Electric field on staggered mesh
	electric_field_staggered= new double[nmesh-1];
	for( int i = 0; i < nmesh; i++){
        	electric_field_staggered[i] = 0.0;
	}
	potential = new double[nmesh];

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
 * Given a point and a (periodic) mesh, return the indices
 * of the grid points either side of x
 */
void Mesh::get_index_pair(const double x, const double *mesh, const int meshsize, int *index_down, int *index_up){

	int index = 1;
	if( x < mesh[0] ){
		*index_down = meshsize-1;
		*index_up   = 0;
	} else {
		while( mesh[index] < x and index < meshsize ){
			index++;
		};
		//std::cout << index << " " << mesh[index]  << "\n";

		*index_down = index - 1;
		if( index == meshsize ){
			*index_up   = 0;
		} else {
			*index_up   = index;
		}
	}
}

/* 
 * Evaluate the electric field at x grid points by interpolating using the
 * values at staggered grid points
 */
double Mesh::evaluate_electric_field(const double x){

	// Implementation of 
        //   np.interp(x,self.half_grid,self.efield)        
	//
	// Find grid cell that x is in
	int index_up, index_down;

	get_index_pair(x, mesh_staggered, nmesh-1, &index_down, &index_up);

	//std::cout << "index : " << index << " nmesh " << nmesh << "\n";
	//std::cout << index_down << " " << index_up  << "\n";
	//std::cout << mesh_staggered[index_down] << " " << x << " " << mesh_staggered[index_up]  << "\n";
	// now x is in the cell ( mesh[index-1], mesh[index] )
	
	double cell_width = mesh_staggered[index_up] - mesh_staggered[index_down];
	// if the cell width is negative, it's because we are in the cell
	// between the upper and lower end of the grid. To get the correct
	// answer, we need to add the domain length on to the cell_width
	if( mesh_staggered[index_up] < mesh_staggered[index_down] ){
		cell_width += 1.0;
	}
	double distance_into_cell = x - mesh_staggered[index_down];
	// similarly, this is only negative if x is in the cell between the
	// upper and lower grid points
	if( distance_into_cell < 0.0 ){
		distance_into_cell += 1.0;
	}

	// r is the proportion if the distance into the cell that the particle is at
	// e.g. midpoint => r = 0.5
	double r = distance_into_cell / cell_width;
        //std::cout << r  << "\n";
	return (1.0 - r) * electric_field_staggered[index_down] + r * electric_field_staggered[index_up];
};

/*
 * Deposit charge at grid points. For a particle in a grid space [a,b],
 * distribute charge at grid points a and b proportionally to the particle's
 * distance from those points.
 */
void Mesh::deposit(Plasma *plasma){

	// Zero the density before depositing
	for(int i = 0; i < nmesh; i++) {
		charge_density[i] = 0.0;
	}

	// Deposite particles
	for(int i = 0; i < plasma->n; i++) {
		// get index of left-hand grid point
		//std::cout << plasma->x[i] << "\n";
		int index = (floor(plasma->x[i]/dx));
		// r is the proportion if the distance into the cell that the particle is at
		// e.g. midpoint => r = 0.5
		double r = plasma->x[i] / dx - double(index);
		//std::cout << r << "\n\n";
		charge_density[index] += (1.0-r) * plasma->w[i]; // / dx;
		charge_density[index+1] += r * plasma->w[i]; // / dx;
	}

	// Ensure result is periodic.
	// The charge index 0 should have contributions from [0,dx] and [1-dx,1],
	// but at this point will only have the [0,dx] contribution. All the
	// [1-dx,1] contribution is at index nmesh-1. To make is periodic we
	// therefore sum the charges at the end points:
	charge_density[0] += charge_density[nmesh-1];
	// Then make the far boundary equal the near boundary
	charge_density[nmesh-1] = charge_density[0];

	//for( int i = 0; i < nmesh; i++){
	//	std::cout << charge_density[i] << "\n";
	//}
}

/*
 * Solve Gauss' law for the electric field using the charge
 * distribution as the RHS. Combine this solve with definition
 * E = - Grad(phi) to do this in a single step.
 */
void Mesh::solve_for_electric_field_fft() {

	FFT f(nintervals);

	// Transform charge density (summed over species)
	for(int i = 0; i < nintervals; i++) {
        	f.in[i][0] = 1.0 - charge_density[i];
        	f.in[i][1] = 0.0;
	}

	fftw_execute(f.plan_forward);

	// Divide by wavenumber
	double tmp; // Working double to allow swap
	for(int i = 0; i < nintervals; i++) {
		// New element = i * poisson_E_factor * old element
		tmp = f.out[i][1];
        	f.out[i][1] = poisson_E_factor[i] * f.out[i][0];
		// Minus to account for factor of i
        	f.out[i][0] = - poisson_E_factor[i] * tmp;
	}
	fftw_execute(f.plan_inverse);

	for(int i = 0; i < nintervals; i++) {
		electric_field[i] = f.in[i][0];
	}
	electric_field[nmesh-1] = electric_field[0];

	fftw_destroy_plan(f.plan_forward);
	fftw_destroy_plan(f.plan_inverse);
	fftw_free(f.in);
	fftw_free(f.out);

}

/*
 * Solve Gauss' law for the electrostatic potential using the charge
 * distribution as the RHS. Take the FFT to diagonalize the problem.
 */
void Mesh::solve_for_potential_fft() {

	FFT f(nintervals);

	// Transform charge density (summed over species)
	for(int i = 0; i < nintervals; i++) {
        	f.in[i][0] = 1.0 - charge_density[i];
        	f.in[i][1] = 0.0;
	}

//	for(int i = 0; i < nmesh; i++) {
//		std::cout << f.in[i] << " ";
//	}
//	std::cout << "\n";
	fftw_execute(f.plan_forward);

//	for(int i = 0; i < nintervals; i++) {
//		std::cout << f.out[i][0] << " " << f.out[i][1] << "\n";
//	}
//	std::cout << "\n";

	// Divide by wavenumber
	for(int i = 0; i < nintervals; i++) {
        	f.out[i][0] *= poisson_factor[i];
        	f.out[i][1] *= poisson_factor[i];
	}
	fftw_execute(f.plan_inverse);

	for(int i = 0; i < nintervals; i++) {
		potential[i] = f.in[i][0];
		//std::cout << potential[i] << " ";
	}
	potential[nmesh-1] = potential[0];
	//std::cout << "\n";

	fftw_destroy_plan(f.plan_forward);
	fftw_destroy_plan(f.plan_inverse);
	fftw_free(f.in);
	fftw_free(f.out);

}

/*
 * Solve Gauss' law for the electrostatic potential using the charge
 * distribution as a solve. In 1D, this is a tridiagonal matrix inversion with
 * the Thomas algorithm.
 */
void Mesh::solve_for_potential() {

	// Initialize with general terms
	for(int i = 0; i < nmesh - 1; i++) {
        	du[i] = 1.0;
        	dl[i] = 1.0;
        	d[i] = -2.0;
        	b[i] = - dx * dx * (charge_density[i] - 1.0 );
	}
        d[nmesh-1] = -2.0;
        b[nmesh-1] = - dx * dx * ( charge_density[nmesh-1] - 1.0 );

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

	// Could save a memcopy here by writing input RHS
	// to potential at top of function
	for(int i = 0; i < nmesh; i++) {
		potential[i] = b[i];
	}
}
        
/*
 * Find the electric field by taking the gradient of the potential
 */
void Mesh::get_electric_field() {
	for( int i = 0; i < nmesh-1; i++){
        	electric_field_staggered[i] = -( potential[i+1] - potential[i] ) / dx;
	}
}
/*
 * Interpolate electric field onto the staggered grid from the unstaggered grid
 */
void Mesh::get_E_staggered_from_E() {
	// E_staggered[i] is halfway between E[i] and E[i+1]
	// NB: No need for nmesh-1 index, as periodic point not kept in
	// electtic_field_staggered
	for( int i = 0; i < nmesh-1; i++){
        	electric_field_staggered[i] = 0.5*(electric_field[i] + electric_field[i+1]);
	}
}
