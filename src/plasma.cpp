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
	x = new double[n]; // particle positions
	for(int i = 0; i < n; i++){
		//x[i] = cos(2.0*M_PI*double(i)/(double(n)-1.0));
		x[i] = cos(std::uniform_real_distribution<double>(0.0,2.0*M_PI)(generator));
	}
	v = new double[n]; // particle velocities
	for(int i = 0; i < n; i++){
		v[i] = std::normal_distribution<double>(0.0,T)(generator);
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
