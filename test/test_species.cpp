#include "../include/species.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(SpeciesTest, Species) {

  Mesh mesh(10);
  Species species(mesh);
  EXPECT_EQ(species.n, 10);
  EXPECT_EQ(species.T, 1.0);
  EXPECT_EQ(species.q, 1);
  EXPECT_EQ(species.m, 1);
  EXPECT_EQ(species.kinetic, true);
  EXPECT_EQ(species.vth, std::sqrt(2.0));

  Species species2(mesh, true, 1.0, -1, 1836, 1);
  EXPECT_EQ(species2.n, 1);
  EXPECT_EQ(species2.T, 1.0);
  EXPECT_EQ(species2.q, -1);
  EXPECT_EQ(species2.m, 1836);
  EXPECT_EQ(species2.kinetic, true);
  EXPECT_EQ(species2.vth, std::sqrt(2.0 / 1836.0));

  Species species3(mesh, true, 3.14159, 2, 9, 34);
  EXPECT_EQ(species3.n, 34);
  EXPECT_EQ(species3.T, 3.14159);
  EXPECT_EQ(species3.q, 2);
  EXPECT_EQ(species3.m, 9);
  EXPECT_EQ(species3.kinetic, true);
  EXPECT_EQ(species3.vth, std::sqrt(2.0 * 3.14159 / 9.0));

  Species species4(mesh, false);
  EXPECT_EQ(species4.T, 1.0);
  EXPECT_EQ(species4.q, 1);
  EXPECT_EQ(species4.m, 1);
  EXPECT_EQ(species4.vth, std::sqrt(2.0));
  EXPECT_EQ(species4.kinetic, false);

  //  for(int i = 0; i < plasma.n; i++){
  //	  std::cout << plasma.x[i] << " ";
  //  	EXPECT_EQ(plasma.x[i], cos(2.0*M_PI*double(i)/(double(plasma.n-1.0))));
  //  }
  //  std::cout << "\n";

  // TODO: test that velocity space has expected
  // properties, probably by taking moments

  for (int i = 0; i < species.n; i++) {
    EXPECT_EQ(species.w[i], 1.0 / (double(species.n)));
  }
}

TEST(SpeciesTest, InitialConditions) {

  // Plasma constructor calls set_initital_conditions
  Mesh mesh(10);
  Species species(mesh);

  // All positions should be in [0,1]
  // All velocities should be in [-6,6]

  for (int i = 0; i < species.n; i++) {
    EXPECT_TRUE(species.x[i] >= 0.0);
    EXPECT_TRUE(species.x[i] <= 1.0);
    EXPECT_TRUE(species.v.x[i] >= -6.0);
    EXPECT_TRUE(species.v.x[i] <= 6.0);
    EXPECT_TRUE(species.v.y[i] == 0.0);
    EXPECT_TRUE(species.v.z[i] == 0.0);
  }
}
