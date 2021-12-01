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
Plasma::Plasma() {

        n = 10; // number of particles
        T = 1.0; // temperature

	x = new double[n]; // particle positions
	v = new double[n]; // particle velocities
	xnew = new double[n]; // particle positions at new timestep
	vnew = new double[n]; // particle velocities at new timestep
}

/*
 * Second order accurate particle pusher
 * with spatially periodic boundary conditions
 */
void Plasma::push(Mesh *mesh) {

	for(int i = 0; i < n; i++) {
         	xnew[i] += 0.5 * mesh->dt * mesh->evaluate_electric_field(x);
         	vnew[i] += mesh->dt * xnew[i];
         	xnew[i] += 0.5 * mesh->dt * mesh->evaluate_electric_field(xnew);

		//apply periodic bcs
                xnew[i] = std::fmod(xnew[i], 1.0);
	}
}
