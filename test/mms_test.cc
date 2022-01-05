#include <gtest/gtest.h>
#include "../src/mesh.hpp"
#include "../src/plasma.hpp"
#include <cmath>

TEST(MMSTest, InitialConditions) {
  Mesh mesh;
  Plasma plasma;

  std::vector<double> nparticles;
  std::vector<double> summed_error;
  double error;

  // Deposit partilces onto mesh, and test that the resulting
  // distribution tends to a cosine.
  int np = 1;
  for(int j = 0; j < 20; j++){
  	np *= 2;
	Plasma plasma(np);

	mesh.deposit(&plasma);
	for(int i = 0; i < mesh.nmesh; i++){
	  std::cout << mesh.charge_density[i] << " ";
	}
	std::cout << "\n";

	error = 0.0;
	for(int i = 0; i < mesh.nmesh; i++){
	  error += std::abs(mesh.charge_density[i] - (1.0 + 0.01 * cos( 2.0*M_PI*mesh.mesh[i]))/double(mesh.nintervals));
	}

	nparticles.push_back(np);
	summed_error.push_back(error);
  }

  // summed error should decay like nparticles^{-1/2}

  for(int j = 0; j < 20; j++){
	  std::cout << summed_error[j] << " ";
  }


  EXPECT_EQ(mesh.t, 0.0);
  EXPECT_EQ(mesh.dt, 0.01);
  EXPECT_EQ(mesh.nt, 1000);
  EXPECT_EQ(mesh.nintervals, 10);
  EXPECT_EQ(mesh.nmesh, 11);
  EXPECT_EQ(mesh.mesh.size(), mesh.nmesh);
  EXPECT_EQ(mesh.dx, 0.1);
  for(int i = 0; i < mesh.mesh.size(); i++){
  	EXPECT_EQ(mesh.mesh.at(i), double(i)*mesh.dx);
  }
  EXPECT_EQ(mesh.mesh_staggered.size(), mesh.nintervals);
  for(int i = 0; i < mesh.mesh_staggered.size(); i++){
  	EXPECT_EQ(mesh.mesh_staggered.at(i), double(i+0.5)*mesh.dx);
  }
}

