class Plasma;

#ifndef __PLASMA_H__
#define __PLASMA_H__

#include "mesh.hpp"

class Velocity {
public:
	// particle velocity array in x direction
	std::vector<double> x;
	// particle velocity array in r direction
	std::vector<double> y;
	// particle velocity array in z direction
	std::vector<double> z;
};


class Plasma {
public:
	Plasma(int n = 10, double T = 1.0);
	// number of particles
    	int n;
	// temperature
	double T;
	// particle position array
	std::vector<double> x;
	// particle velocity structure of arrays
	Velocity v;
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

#endif // __PLASMA_H__
