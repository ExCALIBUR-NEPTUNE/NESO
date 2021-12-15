/*
 * Module for dealing with particles
 */

#include "plasma.hpp"
#include "mesh.hpp"
#include <string>
#include <iostream>
#include <cmath>
#include <random>

/*
 * Initialize particles
 */
Plasma::Plasma(int n_in, double T_in) {

	// Number of particles
	n = n_in;
	// Species temperature
	T = T_in;

	std::default_random_engine generator;

	// Initialize distribution function
	// Pick random triplet (pos, vel, r) and keep particle if r < f(x,v)
	// for f the initial distribution.
	
	int i = 0;
	// trial particle positions and velocities
	double pos, vel, r;
	// amplitude of wave perturbation
	double amp = 0.01; 
	x = new double[n]; // particle positions
	v = new double[n]; // particle velocities
	while( i < n ){
		pos = std::uniform_real_distribution<double>(0.0,1)(generator);
		vel = std::uniform_real_distribution<double>(-6.0,6.0)(generator);
		r = std::uniform_real_distribution<double>(0.0,1.0 + amp)(generator);

		if( r < (1.0 + amp * cos( 2.0*M_PI*pos)) * exp(-vel*vel) ){
			x[i] = pos;
			v[i] = vel;
			i++;
		}
	}

        w = new double[n]; // particle weight
	for(int i = 0; i < n; i++){
		w[i] = 1.0/double(n);
	}
}

/*
 * Second order accurate particle pusher
 * with spatially periodic boundary conditions
 */
void Plasma::push(Mesh *mesh) {

	for(int i = 0; i < n; i++) {
         	v[i] += 0.5 * mesh->dt * mesh->evaluate_electric_field(x[i]);
         	x[i] += mesh->dt * v[i];
         	v[i] += 0.5 * mesh->dt * mesh->evaluate_electric_field(x[i]);

		//apply periodic bcs
		while(x[i] < 0){
			x[i] += 1.0;
		}
                x[i] = std::fmod(x[i], 1.0);
	}
//	for(int i = 0; i < n; i++) {
//		std::cout << x[i] << " ";
//	}
//	std::cout << "\n";
}
