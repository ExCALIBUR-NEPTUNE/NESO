#include <gtest/gtest.h>
#include "src/mesh.hpp"

TEST(PlasmaTest, Mesh) {
  Mesh mesh;
  // Expect equality.
  EXPECT_EQ(mesh.t, 0.0);
  EXPECT_EQ(mesh.dt, 0.01);
  EXPECT_EQ(mesh.nt, 1000);
  EXPECT_EQ(mesh.nintervals, 10);
  EXPECT_EQ(mesh.nmesh, 11);
  EXPECT_EQ(mesh.dx, 0.1);
  for(int i = 0; i < mesh.nmesh-1; i++){
  	EXPECT_EQ(mesh.mesh[i], double(i)*mesh.dx);
  }
  for(int i = 0; i < mesh.nmesh-2; i++){
  	EXPECT_EQ(mesh.mesh_staggered[i], double(i+0.5)*mesh.dx);
  }
}

TEST(PlasmaTest, get_index_pair) {
  Mesh mesh;
  double x;
  for(int i = 0; i < mesh.nmesh-1; i++){
	  std::cout << mesh.mesh_staggered[i] << " ";
  }
  std::cout << "\n";

  int index_down, index_up;
  int index_down_expected, index_up_expected;

  for(int i = 0; i < mesh.nmesh; i++){
	x = double(i)/double(mesh.nmesh-1);
	std::cout << x << "\n";

  	mesh.get_index_pair(x,mesh.mesh_staggered,mesh.nmesh-1,&index_down,&index_up);
  	std::cout << index_down << " " << index_up  << "\n";

	index_down_expected = i-1;
	index_up_expected = i;
	if(index_down_expected < 0){
		index_down_expected = mesh.nmesh-2;
	}
	if(index_up_expected > mesh.nmesh-2){
		index_up_expected = 0;
	}

  	EXPECT_EQ(index_down, index_down_expected);
  	EXPECT_EQ(index_up, index_up_expected);
  }
}

TEST(PlasmaTest, evaluate_electric_field) {
  Mesh mesh;
  //for(int i = 0; i < mesh.nmesh-1; i++){
  //	  std::cout << mesh.mesh_staggered[i] << " ";
  //}
  //std::cout << "\n";

  // mock up electric field to interpolate
  for(int i = 0; i < mesh.nmesh-1; i++){
	  mesh.electric_field_staggered[i] = double(i);
	  //std::cout << mesh.electric_field_staggered[i] << " ";
  }
  //std::cout << "\n";

  // Test selection of points:

  // below lowest point
  double x = 0.0;
  double E = mesh.evaluate_electric_field(x);
  ASSERT_NEAR(E, 4.5, 1e-8); // midpoint between 0 and 9

  x = 0.075;
  E = mesh.evaluate_electric_field(x);
  ASSERT_DOUBLE_EQ(E, 0.25); // 0.75 * 0 + 0.25 * 1

  x = 0.25;
  E = mesh.evaluate_electric_field(x);
  ASSERT_DOUBLE_EQ(E, 2); // on grid point 2

  x = 0.6;
  E = mesh.evaluate_electric_field(x);
  ASSERT_DOUBLE_EQ(E, 5.5); // midpoint between 5 and 6

  x = 0.975;
  E = mesh.evaluate_electric_field(x);
  ASSERT_NEAR(E, 6.75, 1e-8); // 0.75*9 + 0.25*0
}
