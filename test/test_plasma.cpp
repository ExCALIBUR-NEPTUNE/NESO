#include <gtest/gtest.h>
#include "plasma.hpp"
#include <cmath>

TEST(PlasmaTest, Plasma) {

  Plasma plasma;
  EXPECT_EQ(plasma.n, 10);
  EXPECT_EQ(plasma.T, 1.0);

  Plasma plasma2(1,1.0);
  EXPECT_EQ(plasma2.n, 1);
  EXPECT_EQ(plasma2.T, 1.0);

  Plasma plasma3(34,3.14159);
  EXPECT_EQ(plasma3.n, 34);
  EXPECT_EQ(plasma3.T, 3.14159);

//  for(int i = 0; i < plasma.n; i++){
//	  std::cout << plasma.x[i] << " ";
//  	EXPECT_EQ(plasma.x[i], cos(2.0*M_PI*double(i)/(double(plasma.n-1.0))));
//  }
//  std::cout << "\n";

  // TODO: test that velocity space has expected
  // properties, probably by taking moments
  
  for(int i = 0; i < plasma.n; i++){
  	EXPECT_EQ(plasma.w[i], 1.0/(double(plasma.n)));
  }
}

TEST(PlasmaTest, InitialConditions) {

	// Plasma constructor calls set_initital_conditions
	Plasma plasma;

	// All positions should be in [0,1]
	// All velocities should be in [-6,6]

	for(int i = 0; i < plasma.n; i++){
		EXPECT_TRUE( plasma.x[i] >= 0.0 );
		EXPECT_TRUE( plasma.x[i] <= 1.0 );
		EXPECT_TRUE( plasma.v[i] >= -6.0 );
		EXPECT_TRUE( plasma.v[i] <= 6.0 );
	}
}
