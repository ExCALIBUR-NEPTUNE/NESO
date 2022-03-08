class Diagnostics;

#ifndef __DIAGNOSTICS_H__
#define __DIAGNOSTICS_H__

#include "mesh.hpp"
#include "plasma.hpp"

class Diagnostics {
public:
	// Vector of times at each timestep
	std::vector<double> time;
	// Total energy at each timestep
	std::vector<double> total_energy;
	// Total particle kinetic energy at each timestep
	std::vector<double> particle_energy;
	// Total energy in the electric field at each timestep
	std::vector<double> field_energy;

	// Store simulation time as a vector
	void store_time(const double t);
	// Compute the total energy at a timestep
	void compute_total_energy(Mesh &mesh, Plasma &plasma);
	// Compute the energy in the electric field at a timestep
	void compute_field_energy(Mesh &mesh);
	// Compute the total kinetic energy of particles at a timestep
	void compute_particle_energy(const Plasma &plasma);
};

#endif // __DIAGNOSTICS_H__
