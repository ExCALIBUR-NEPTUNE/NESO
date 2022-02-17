class Species;

#ifndef __SPECIES_H__
#define __SPECIES_H__

#include "mesh.hpp"
#include "velocity.hpp"

class Species {
public:
	Species(int n = 10, double T = 1.0, int q = 1, bool adiabatic = false);
	// Whether this species is treated adiabatically
	bool adiabatic;
	// number of particles
    	int n;
	// temperature
	double T;
	// charge
	int q;
	// particle position array
	std::vector<double> x;
	// particle velocity structure of arrays
	Velocity v;
	// charge density of species (if adiabatic)
	double charge_density;
	// particle position array at
	// next timestep
	std::vector<double> xnew;
	// particle velocity array at
	// next tmiestep
	std::vector<double> vnew;
	// particle weight
	std::vector<double> w;
	// particle pusher
	void push(Mesh *mesh);
	// initial conditions 
	void set_initial_conditions(std::vector<double> &x, Velocity &v);
};

#endif // __SPECIES_H__
