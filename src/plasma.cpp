/*
 * Module for dealing with particles
 */

#include "plasma.hpp"
#include "mesh.hpp"
#include "species.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

/*
 * Initialize particles
 */
Plasma::Plasma(std::vector<Species> species_list) {

  nspec = species_list.size();
  n_kinetic_spec = 0;
  n_adiabatic_spec = 0;

  for (int i = 0; i < nspec; i++) {
    species.push_back(species_list.at(i));
    if (species_list.at(i).kinetic) {
      kinetic_species.push_back(species_list.at(i));
      n_kinetic_spec++;
    } else {
      adiabatic_species.push_back(species_list.at(i));
      n_adiabatic_spec++;
    }
  }
}

/*
 * Second order accurate particle pusher
 * with spatially periodic boundary conditions
 */
void Plasma::push(sycl::queue &q, Mesh &mesh) {

  for (int i = 0; i < n_kinetic_spec; i++) {
    kinetic_species.at(i).push(q, &mesh);
  }
}
