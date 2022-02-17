#include <gtest/gtest.h>
#include "../src/plasma.hpp"
#include <cmath>

TEST(PlasmaTest, Plasma) {

  Species electrons(true);
  Species ions(false);
  Species neutrals(2,true);
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

}
