/*
 * Module for dealing with particles
 */

#include "mesh.hpp"
#include "plasma.hpp"
#include "fft.hpp"
#include <string>
#include <iostream>
#include <cmath>
#include <vector>
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
	mesh.resize(nmesh);
	for( int i = 0; i < mesh.size(); i++){
        	mesh.at(i) = double(i)*dx;
	}
	// mesh point vector staggered at half points
	mesh_staggered.resize(nintervals);
	for( int i = 0; i < mesh_staggered.size(); i++){
        	mesh_staggered.at(i) = double(i+0.5)*dx;
	}
	// Fourier wavenumbers k[j] =
	// 2*pi*j such that k*mesh gives
	// the argument of the exponential
	// in FFTW
	// NB: for complex to complex transforms, the second half of Fourier
	// modes must be negative of the first half
	k.resize(nmesh);
	for( int i = 0; i < (k.size()/2)+1; i++){
        	k.at(i) = 2.0*M_PI*double(i);
        	k.at(k.size()-i-1) = -k.at(i);
	}
	// Poisson factor
	// Coefficient to multiply Fourier-transformed charge density by in the
	// Poisson solve:
	// - 1 / (k**2 * nmesh)
	// This accounts for the change of sign, the wavenumber squared (for
	// the Laplacian), and the length of the array (the normalization in
	// the FFT).
	poisson_factor.resize(nintervals);
        poisson_factor.at(0) = 0.0;
	for( int i = 1; i < poisson_factor.size(); i++){
        	poisson_factor.at(i) = -1.0/(k.at(i)*k.at(i)*double(nintervals));
	}

	// Poisson Electric field factor
	// Coefficient to multiply Fourier-transformed charge density by in the
	// Poisson solve:
	// i / (k * nmesh)
	// to obtain the electric field from the Poisson equation and 
	// E = - Grad(phi) in a single step.
	// This accounts for the change of sign, the wavenumber squared (for
	// the Laplacian), the length of the array (the normalization in the
	// FFT), and the factor of (-ik) for taking the Grad in fourier space.
	// NB Rather than deal with complex arithmetic, we make this a real
	// number that we apply to the relevant array entry.
	poisson_E_factor.resize(nintervals);
        poisson_E_factor.at(0) = 0.0;
	for( int i = 1; i < poisson_E_factor.size(); i++){
        	poisson_E_factor.at(i) = 1.0/(k.at(i)*double(nintervals));
	}

	charge_density.resize(nmesh);
	for( int i = 0; i < charge_density.size(); i++){
        	charge_density.at(i) = 0.0;
	}
	// Electric field on mesh
	electric_field.resize(nmesh);
	for( int i = 0; i < electric_field.size(); i++){
        	electric_field.at(i) = 0.0;
	}
	// Electric field on staggered mesh
	electric_field_staggered.resize(nintervals);
	for( int i = 0; i < electric_field_staggered.size(); i++){
        	electric_field_staggered.at(i) = 0.0;
	}
	potential.resize(nmesh);

	// NB these must be double * for use in lapack call
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
 * Given a point and a mesh, return the index of the grid point immediately to
 * the left of x. When we include both endpoints on the mesh, the point x is
 * always between the grid's upper and lower bounds, so the we only need the
 * left-hand index
 */
int Mesh::get_left_index(const double x, const std::vector<double> mesh){

	int index = 0;
	while( mesh.at(index+1) < x and index < mesh.size() ){
		index++;
	};
	return index;
}

/* 
 * Evaluate the electric field at x grid points by
 * interpolating onto the grid
 */
double Mesh::evaluate_electric_field(const double x){

	// Implementation of 
        //   np.interp(x,self.half_grid,self.efield)        
	//
	// Find grid cell that x is in
	int index = get_left_index(x, mesh);

	//std::cout << "index : " << index << " nmesh " << nmesh << "\n";
	//std::cout << index_down << " " << index_up  << "\n";
	//std::cout << mesh_staggered[index_down] << " " << x << " " << mesh_staggered[index_up]  << "\n";
	// now x is in the cell ( mesh[index-1], mesh[index] )
	
	double cell_width = mesh.at(index+1) - mesh.at(index);
	double distance_into_cell = x - mesh.at(index);

	// r is the proportion if the distance into the cell that the particle is at
	// e.g. midpoint => r = 0.5
	double r = distance_into_cell / cell_width;
        //std::cout << r  << "\n";
	return (1.0 - r) * electric_field.at(index) + r * electric_field.at(index+1);
};

/*
 * Deposit charge at grid points. For a particle in a grid space [a,b],
 * distribute charge at grid points a and b proportionally to the particle's
 * distance from those points.
 */
void Mesh::deposit(Plasma *plasma){

	// Zero the density before depositing
	for(int i = 0; i < charge_density.size(); i++) {
		charge_density.at(i) = 0.0;
	}

	// Deposite particles
	for(int i = 0; i < plasma->n; i++) {
		// get index of left-hand grid point
		//std::cout << plasma->x[i] << "\n";
		int index = (floor(plasma->x.at(i)/dx));
		// r is the proportion if the distance into the cell that the particle is at
		// e.g. midpoint => r = 0.5
		double r = plasma->x.at(i) / dx - double(index);
		//std::cout << r << "\n\n";
		charge_density.at(index) += (1.0-r) * plasma->w.at(i);
		charge_density.at(index+1) += r * plasma->w.at(i);
	}

	// Ensure result is periodic.
	// The charge index 0 should have contributions from [0,dx] and [1-dx,1],
	// but at this point will only have the [0,dx] contribution. All the
	// [1-dx,1] contribution is at index nmesh-1. To make is periodic we
	// therefore sum the charges at the end points:
	charge_density.at(0) += charge_density.at(charge_density.size()-1);
	// Then make the far boundary equal the near boundary
	charge_density.at(charge_density.size()-1) = charge_density.at(0);

	//for( int i = 0; i < nmesh; i++){
	//	std::cout << charge_density[i] << "\n";
	//}
}

/*
 * Solve Gauss' law for the electric field using the charge
 * distribution as the RHS. Combine this solve with definition
 * E = - Grad(phi) to do this in a single step.
 */
void Mesh::solve_for_electric_field_fft(FFT *f) {

	// Transform charge density (summed over species)
	for(int i = 0; i < nintervals; i++) {
        	f->in[i][0] = 1.0 - charge_density.at(i);
        	f->in[i][1] = 0.0;
	}

	fftw_execute(f->plan_forward);

	// Divide by wavenumber
	double tmp; // Working double to allow swap
	for(int i = 0; i < poisson_E_factor.size(); i++) {
		// New element = i * poisson_E_factor * old element
		tmp = f->out[i][1];
        	f->out[i][1] = poisson_E_factor.at(i) * f->out[i][0];
		// Minus to account for factor of i
        	f->out[i][0] = - poisson_E_factor.at(i) * tmp;
	}
	fftw_execute(f->plan_inverse);

	for(int i = 0; i < electric_field.size()-1; i++) {
		electric_field.at(i) = f->in[i][0];
	}
	electric_field.at(electric_field.size()-1) = electric_field.at(0);

}

/*
 * Solve Gauss' law for the electrostatic potential using the charge
 * distribution as the RHS. Take the FFT to diagonalize the problem.
 */
void Mesh::solve_for_potential_fft(FFT *f) {

	// Transform charge density (summed over species)
	for(int i = 0; i < nintervals; i++) {
        	f->in[i][0] = 1.0 - charge_density[i];
        	f->in[i][1] = 0.0;
	}

//	for(int i = 0; i < nmesh; i++) {
//		std::cout << f.in[i] << " ";
//	}
//	std::cout << "\n";
	fftw_execute(f->plan_forward);

//	for(int i = 0; i < nintervals; i++) {
//		std::cout << f.out[i][0] << " " << f.out[i][1] << "\n";
//	}
//	std::cout << "\n";

	// Divide by wavenumber
	for(int i = 0; i < nintervals; i++) {
        	f->out[i][0] *= poisson_factor[i];
        	f->out[i][1] *= poisson_factor[i];
	}
	fftw_execute(f->plan_inverse);

	for(int i = 0; i < nintervals; i++) {
		potential[i] = f->in[i][0];
		//std::cout << potential[i] << " ";
	}
	potential[nmesh-1] = potential[0];
	//std::cout << "\n";

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
