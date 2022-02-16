class Plasma;

#ifndef __PLASMA_H__
#define __PLASMA_H__

#include "mesh.hpp"
#include "species.hpp"

class Plasma {
public:
	Plasma(std::vector<Species> species_list);
	// list of all species in the plasma
	std::vector<Species> species;
	// list of all kinetic species in the plasma
	// this allows us to perform loops without conditionals on whether a species is kinetic
	std::vector<Species> kinetic_species;
	// number of species in
	// the plasma
	int nspec;
	// number of kinetic
	// species in the plasma
	int nkineticspec;

	// particle pusher
	void push(Mesh *mesh);
};

#endif // __PLASMA_H__
