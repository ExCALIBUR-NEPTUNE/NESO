class Plasma;

#ifndef __PLASMA_H__
#define __PLASMA_H__

#include "mesh.hpp"
#include "species.hpp"

class Plasma {
public:
	Plasma(std::vector<Species> species_list);
	// particle position array
	std::vector<Species> species;
	// number of species in
	// the plasma
	int nspec;

	// particle pusher
	void push(Mesh *mesh);
};

#endif // __PLASMA_H__
