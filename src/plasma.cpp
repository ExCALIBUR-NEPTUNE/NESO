/*
 * Module for dealing with particles
 */

#include "plasma.hpp"
#include "species.hpp"
#include "mesh.hpp"
#include <string>
#include <iostream>
#include <cmath>
#include <random>

/*
 * Initialize particles
 */
Plasma::Plasma(std::vector<Species> species_list) {

	nspec = species_list.size();
	n_kinetic_spec = 0;
	n_adiabatic_spec = 0;

	for(int i = 0; i < nspec; i++){
		species.push_back(species_list.at(i));
		if( species_list.at(i).kinetic ){
			kinetic_species.push_back(species_list.at(i));
			n_kinetic_spec++;
		}
		else {
			adiabatic_species.push_back(species_list.at(i));
			n_adiabatic_spec++;
		}
	}
}

/*
 * Second order accurate particle pusher
 * with spatially periodic boundary conditions
 */
void Plasma::push(Mesh &mesh) {

	for(int i = 0; i < n_kinetic_spec; i++) {
		kinetic_species.at(i).push(mesh);
	}
}
