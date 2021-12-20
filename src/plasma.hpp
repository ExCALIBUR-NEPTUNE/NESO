class Plasma;

#ifndef __PLASMA_H__
#define __PLASMA_H__

#include "mesh.hpp"

class Plasma {
public:
	Plasma(int n = 10, double T = 1.0);
	// number of particles
    	int n;
	// temperature
	double T;
	// particle position array
	double *x;
	// particle velocity array
	double *v;
	// particle position array at
	// next timestep
	double *xnew;
	// particle velocity array at
	// next tmiestep
	double *vnew;
	// particle weight
        double *w;
	// particle pusher
	void push(Mesh *mesh);
	// initial conditions 
	void set_initial_conditions(double *x, double *v);
};

#endif // __PLASMA_H__
