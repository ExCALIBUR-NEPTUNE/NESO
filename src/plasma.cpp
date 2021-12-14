/*
 * Module for dealing with particles
 */

#include "plasma.hpp"
#include "mesh.hpp"
#include <string>
#include <iostream>
#include <cmath>

/*
 * Initialize particles
 */
Plasma::Plasma(int n_in, double T_in) {

	// Number of particles
	n = n_in;
	// Species temperature
	T = T_in;

	x = new double[n]; // particle positions
	for(int i = 0; i < n; i++){
		x[i] = double(i)/(double(n)-1.0);
	}
	v = new double[n]; // particle velocities
	xnew = new double[n]; // particle positions at new timestep
	vnew = new double[n]; // particle velocities at new timestep

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
         	xnew[i] += 0.5 * mesh->dt * mesh->evaluate_electric_field(x[i]);
         	vnew[i] += mesh->dt * xnew[i];
         	xnew[i] += 0.5 * mesh->dt * mesh->evaluate_electric_field(xnew[i]);

		//apply periodic bcs
                xnew[i] = std::fmod(xnew[i], 1.0);
	}
}
