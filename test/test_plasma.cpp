#include <gtest/gtest.h>
#include "plasma.hpp"
#include <cmath>

TEST(PlasmaTest, Plasma) {

  Mesh mesh(10);
  Species electrons(mesh,true);
  Species ions(mesh,false);
  Species neutrals(mesh,true,1.0,0,1836,2);
  std::vector<Species> species_list;
  species_list.push_back(electrons);
  species_list.push_back(ions);
  species_list.push_back(neutrals);
  Plasma plasma(species_list);

  EXPECT_EQ(plasma.nspec, 3);
  EXPECT_EQ(plasma.n_kinetic_spec, 2);
  EXPECT_EQ(plasma.n_adiabatic_spec, 1);

  // More rigorous testing would be good, but test that
  // the correct values are set in certain places
  EXPECT_EQ(plasma.kinetic_species.at(1).n, 2);
  EXPECT_EQ(plasma.kinetic_species.at(1).q, 0);

}
