/*
 * Module for dealing with particles
 */

#include "species.hpp"
#include "mesh.hpp"
#include <string>
#include <iostream>
#include <cmath>
#include <random>

/*
 * Initialize particles
 */
Species::Species(int n_in, double T_in, int q_in, bool adiabatic_in) {

	// Whether this species is treated adiabatically (true) or kinnetically (flase)
	adiabatic = adiabatic_in;
	// Number of particles
	n = n_in;
	// Species temperature
	T = T_in;
	// Species charge
	q = q_in;

	x.resize(n); // particle positions
	v.x.resize(n); // particle velocities
	v.y.resize(n); // particle velocities
	v.z.resize(n); // particle velocities
	
	set_initial_conditions(x, v);

        w.resize(n); // particle weight
	for(int i = 0; i < n; i++){
		w[i] = 1.0/double(n);
	}
}

/*
 * Initialize distribution function
 * Pick random triplet (pos, vel, r) and keep particle if r < f(x,v)
 * for f the initial distribution.
 */
void Species::set_initial_conditions(std::vector<double> &x, Velocity &v) {

	// trial particle positions and velocities
	double pos, vel, r;
	// amplitude of wave perturbation
	double amp = 1e-8; 
	double big = 1e8;

	int i = 0;
	std::default_random_engine generator;
	while( i < n ){
		pos = std::uniform_real_distribution<double>(0.0,1)(generator);
		vel = std::uniform_real_distribution<double>(-6.0,6.0)(generator);
		r = std::uniform_real_distribution<double>(0.0,1.0)(generator);

		//if( r * (1.0 + amp) < (1.0 + amp * cos( 2.0*M_PI*pos)) * exp(-vel*vel) / sqrt(M_PI) ){
		//if( r * (1.0 + amp) < (1.0 + amp * cos( 2.0*M_PI*pos)) * exp(-vel*vel) ){
		//if( r  <  amp * cos( 2.0*M_PI*pos) * exp(-vel*vel) / sqrt(M_PI) ) {
		//if( r < exp(-vel*vel) ) {
		//if( r < 0.5 * big * ( exp(- big*(vel-0.5)*(vel-0.5)) + exp(- big*(vel+0.5)*(vel+0.5)) )) {
		if( r < 0.5 * big * ( exp(- big*(vel-1.0)*(vel-1.0)) + exp(- big*(vel+1.0)*(vel+1.0)) )) {
			x.at(i) = pos;
			v.x.at(i) = vel;
			v.y.at(i) = 0.0;
			v.z.at(i) = 0.0;
			i++;
		}
	}
}

/*
 * Second order accurate particle pusher
 * with spatially periodic boundary conditions
 */
void Species::push(Mesh *mesh) {

	for(int i = 0; i < n; i++) {
         	v.x.at(i) += 0.5 * mesh->dt * mesh->evaluate_electric_field(x.at(i));
         	x.at(i) += mesh->dt * v.x.at(i);

		//apply periodic bcs
		while(x.at(i) < 0){
			x.at(i) += 1.0;
		}
                x.at(i) = std::fmod(x.at(i), 1.0);

         	v.x.at(i) += 0.5 * mesh->dt * mesh->evaluate_electric_field(x.at(i));

	}
}
