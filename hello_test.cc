#include <gtest/gtest.h>
#include "src/mesh.hpp"

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(HelloTest, Mesh) {
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

TEST(HelloTest, evaluate_electric_field) {
  Mesh mesh;
  double x = 0.6;
  for(int i = 0; i < mesh.nmesh-2; i++){
	  std::cout << mesh.mesh_staggered[i] << " ";
  }
  std::cout << "\n";

  // mock up electric field to interpolate
  for(int i = 0; i < mesh.nmesh-2; i++){
	  mesh.electric_field_staggered[i] = double(i);
	  std::cout << mesh.electric_field_staggered[i] << " ";
  }
  std::cout << "\n";
  double E = mesh.evaluate_electric_field(x);

  // Expect equality.
  EXPECT_EQ(E, 5.5);
}
