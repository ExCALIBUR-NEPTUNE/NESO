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
Plasma::Plasma(std::vector<Species> species_list) {

	//set_initial_conditions(x, v);

}

/*
 * Initialize distribution function
 * Pick random triplet (pos, vel, r) and keep particle if r < f(x,v)
 * for f the initial distribution.
 */
//void Plasma::set_initial_conditions(std::vector<double> &x, Velocity &v) {
//
//	// trial particle positions and velocities
//	double pos, vel, r;
//	// amplitude of wave perturbation
//	double amp = 1e-8; 
//	double big = 1e8;
//
//	int i = 0;
//	std::default_random_engine generator;
//	while( i < n ){
//		pos = std::uniform_real_distribution<double>(0.0,1)(generator);
//		vel = std::uniform_real_distribution<double>(-6.0,6.0)(generator);
//		r = std::uniform_real_distribution<double>(0.0,1.0)(generator);
//
//		//if( r * (1.0 + amp) < (1.0 + amp * cos( 2.0*M_PI*pos)) * exp(-vel*vel) / sqrt(M_PI) ){
//		//if( r * (1.0 + amp) < (1.0 + amp * cos( 2.0*M_PI*pos)) * exp(-vel*vel) ){
//		//if( r  <  amp * cos( 2.0*M_PI*pos) * exp(-vel*vel) / sqrt(M_PI) ) {
//		//if( r < exp(-vel*vel) ) {
//		//if( r < 0.5 * big * ( exp(- big*(vel-0.5)*(vel-0.5)) + exp(- big*(vel+0.5)*(vel+0.5)) )) {
//		if( r < 0.5 * big * ( exp(- big*(vel-1.0)*(vel-1.0)) + exp(- big*(vel+1.0)*(vel+1.0)) )) {
//			x.at(i) = pos;
//			v.x.at(i) = vel;
//			v.y.at(i) = 0.0;
//			v.z.at(i) = 0.0;
//			i++;
//		}
//	}
//}

/*
 * Second order accurate particle pusher
 * with spatially periodic boundary conditions
 */
void Plasma::push(Mesh *mesh) {

	for(int i = 0; i < nspec; i++) {
		species.at(i).push(mesh);
	}
}
