/*
 * Module for dealing with diagnostics
 */

#include "plasma.hpp"
#include "mesh.hpp"
#include "diagnostics.hpp"
//#include <string>
//#include <iostream>
#include <cmath>


/*
 * Store simulation time as a vector
 */
void Diagnostics::store_time(double t){
	time.push_back(t);
}

/*
 * Compute and store total energy
 */
void Diagnostics::compute_total_energy(Mesh *mesh, Plasma *plasma){

	compute_field_energy(mesh);
	compute_particle_energy(plasma);

	total_energy.push_back(field_energy.back() + particle_energy.back());

}

/*
 * Compute and store the energy in the electric field
 */
void Diagnostics::compute_field_energy(Mesh *mesh) {

	double energy = 0.0;
	// NB: Omit periodic point
	for( std::size_t i = 0; i < mesh->electric_field.size()-1; i++) {
		energy += std::pow(mesh->electric_field.at(i),2);
	}
	energy *= 0.5 / std::pow(mesh->normalized_box_length,2);

	field_energy.push_back(energy);
}

/*
 * Compute and store the energy in the particles
 */
void Diagnostics::compute_particle_energy(Plasma *plasma) {

	double energy = 0.0;
	for( std::size_t j = 0; j < plasma->nkineticspec; j++) {
		for( std::size_t i = 0; i < plasma->kinetic_species.at(j).n; i++) {
			energy += plasma->kinetic_species.at(j).w.at(i)*(
					std::pow(plasma->kinetic_species.at(j).v.x.at(i),2)
					+ std::pow(plasma->kinetic_species.at(j).v.y.at(i),2)
					+ std::pow(plasma->kinetic_species.at(j).v.z.at(i),2)
				);
		}
	}
	energy *= 0.5;

	particle_energy.push_back(energy);
}
