class Plasma;

#ifndef __PLASMA_H__
#define __PLASMA_H__

#include "mesh.hpp"
#include "species.hpp"

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

class Plasma {
public:
	Plasma(std::vector<Species> species_list);
	// list of all species in the plasma
	std::vector<Species> species;
	// list of all kinetic species in the plasma
	// this allows us to perform loops without conditionals on whether a species is kinetic
	std::vector<Species> kinetic_species;
	// list of all adiiabatic species in the plasma
	std::vector<Species> adiabatic_species;
	// number of species in
	// the plasma
	int nspec;
	// number of kinetic
	// species in the plasma
	int n_kinetic_spec;
	// number of adiabatic
	// species in the plasma
	int n_adiabatic_spec;

	// particle pusher
	void push(sycl::queue &q, Mesh *mesh);
};

#endif // __PLASMA_H__
