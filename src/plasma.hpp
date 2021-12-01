class Plasma;

#ifndef __PLASMA_H__
#define __PLASMA_H__

#include "mesh.hpp"

class Plasma {
public:
	Plasma();
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
	// particle pusher
	void push(Mesh *mesh);
};

#endif // __PLASMA_H__
