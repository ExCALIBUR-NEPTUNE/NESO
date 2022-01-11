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
	for( std::size_t i = 0; i < mesh->electric_field.size(); i++) {
		energy += std::pow(mesh->electric_field.at(i),2);
	}

	field_energy.push_back(energy);
}

/*
 * Compute and store the energy in the particles
 */
void Diagnostics::compute_particle_energy(Plasma *plasma) {

	double energy = 0.0;
	for( std::size_t i = 0; i < plasma->n; i++) {
		energy += plasma->w.at(i)*0.5*std::pow(plasma->v.at(i),2);
	}

	particle_energy.push_back(energy);
}
