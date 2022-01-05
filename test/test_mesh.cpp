#include <gtest/gtest.h>
#include "mesh.hpp"
#include "plasma.hpp"
#include <cmath>

TEST(MeshTest, Mesh) {
  Mesh mesh;
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

TEST(MeshTest, get_left_index) {
  Mesh mesh;
  double x;

  int index;
  int index_expected;

  for(int i = 0; i < mesh.mesh.size()-1; i++){
	x = double(i+1e-8)/double(mesh.mesh.size()-1);

  	index = mesh.get_left_index(x,mesh.mesh);

	index_expected = i;

  	EXPECT_EQ(index, index_expected);
  }

  for(int i = 0; i < mesh.mesh.size()-1; i++){
	x = double(i+1.0-1e-8)/double(mesh.mesh.size()-1);

  	index = mesh.get_left_index(x,mesh.mesh);

	index_expected = i;

  	EXPECT_EQ(index, index_expected);
  }

  for(int i = 0; i < mesh.mesh.size()-1; i++){
	x = double(i+0.2)/double(mesh.mesh.size()-1);

  	index = mesh.get_left_index(x,mesh.mesh);

	index_expected = i;

  	EXPECT_EQ(index, index_expected);
  }
}

TEST(MeshTest, evaluate_electric_field) {
  Mesh mesh;
  //for(int i = 0; i < mesh.nmesh-1; i++){
  //	  std::cout << mesh.mesh_staggered[i] << " ";
  //}
  //std::cout << "\n";

  // mock up electric field to interpolate
  for(int i = 0; i < mesh.mesh.size(); i++){
	  mesh.electric_field[i] = double(i);
	  //std::cout << mesh.electric_field_staggered[i] << " ";
  }
  //std::cout << "\n";

  // Test selection of points:

  // at lowest point
  double x = 0.0;
  double E = mesh.evaluate_electric_field(x);
  ASSERT_NEAR(E, 0.0, 1e-8);

  x = 0.075;
  E = mesh.evaluate_electric_field(x);
  ASSERT_DOUBLE_EQ(E, 0.75); // 0.75 * 1 + 0.25 * 0

  x = 0.25;
  E = mesh.evaluate_electric_field(x);
  ASSERT_DOUBLE_EQ(E, 2.5); // halfway between 2 and 3

  x = 0.6;
  E = mesh.evaluate_electric_field(x);
  ASSERT_DOUBLE_EQ(E, 6.0); // on grid point

  x = 0.975;
  E = mesh.evaluate_electric_field(x);
  ASSERT_NEAR(E, 9.75, 1e-8); // 0.75*9 + 0.25*8
}



TEST(MeshTest, deposit) {
  Mesh mesh;
  // Single particle plasma
  Plasma plasma(1,1.0);

  // Single particle at midpoint between first two grid points
  plasma.x[0] = 0.05;
  mesh.deposit(&plasma);
  ASSERT_NEAR(mesh.charge_density[0], 0.5, 1e-8);
  ASSERT_NEAR(mesh.charge_density[1], 0.5, 1e-8);
  for(int i = 2; i < mesh.nmesh-1; i++){
    ASSERT_NEAR(mesh.charge_density[i], 0.0, 1e-8);
  }
  ASSERT_NEAR(mesh.charge_density[mesh.nmesh-1], 0.5, 1e-8);

  double total_charge = 0.0;
  for(int i = 0; i < mesh.nmesh-1; i++){ // Skip repeat point
	  total_charge += mesh.charge_density[i];
  }
  ASSERT_NEAR(total_charge, 1.0, 1e-8);


  plasma.x[0] = 0.5;
  mesh.deposit(&plasma);
  for(int i = 0; i < mesh.nmesh; i++){
	  if(i == 5){
    		ASSERT_NEAR(mesh.charge_density[i], 1.0, 1e-8);
	  } else {
		ASSERT_NEAR(mesh.charge_density[i], 0.0, 1e-8);
	  }
  }
  total_charge = 0.0;
  for(int i = 0; i < mesh.nmesh-1; i++){ // Skip repeat point
	  total_charge += mesh.charge_density[i];
  }
  ASSERT_NEAR(total_charge, 1.0, 1e-8);

  plasma.x[0] = 0.925;
  mesh.deposit(&plasma);
  ASSERT_NEAR(mesh.charge_density[0], 0.25, 1e-8);
  for(int i = 1; i < mesh.nmesh-2; i++){
	ASSERT_NEAR(mesh.charge_density[i], 0.0, 1e-8);
  }
  ASSERT_NEAR(mesh.charge_density[mesh.nmesh-2], 0.75, 1e-8);
  ASSERT_NEAR(mesh.charge_density[mesh.nmesh-1], 0.25, 1e-8);
  total_charge = 0.0;
  for(int i = 0; i < mesh.nmesh-1; i++){ // Skip repeat point
	  total_charge += mesh.charge_density[i];
  }
  ASSERT_NEAR(total_charge, 1.0, 1e-8);

  // Two particle plasma
  Plasma plasma2(2,1.0);

  // Single particle at midpoint between first two grid points
  plasma2.x[0] = 0.05;
  plasma2.x[1] = 0.1;
  mesh.deposit(&plasma2);
  ASSERT_NEAR(mesh.charge_density[0], 0.25, 1e-8);
  ASSERT_NEAR(mesh.charge_density[1], 0.75, 1e-8);
  for(int i = 2; i < mesh.nmesh-1; i++){
    ASSERT_NEAR(mesh.charge_density[i], 0.0, 1e-8);
  }
  ASSERT_NEAR(mesh.charge_density[mesh.nmesh-1], 0.25, 1e-8);

  total_charge = 0.0;
  for(int i = 0; i < mesh.nmesh-1; i++){ // Skip repeat point
	  total_charge += mesh.charge_density[i];
  }
  ASSERT_NEAR(total_charge, 1.0, 1e-8);

}

TEST(MeshTest, get_electric_field) {
  Mesh mesh;

  for(int i = 0; i < mesh.nmesh; i++){
  	  mesh.potential[i] = 0.0;
  }
  mesh.get_electric_field();
  for(int i = 0; i < mesh.nmesh-1; i++){
    ASSERT_NEAR(mesh.electric_field_staggered[i], 0.0, 1e-8);
  }

  for(int i = 0; i < mesh.nmesh; i++){
  	  mesh.potential[i] = double(i);
  }
  mesh.get_electric_field();
  for(int i = 0; i < mesh.nmesh-1; i++){
    ASSERT_NEAR(mesh.electric_field_staggered[i], -1.0/mesh.dx, 1e-8);
  }

  for(int i = 0; i < mesh.nmesh; i++){
  	  mesh.potential[i] = double(i*i);
  }
  mesh.get_electric_field();
  for(int i = 0; i < mesh.nmesh-1; i++){
	  ASSERT_NEAR(mesh.electric_field_staggered[i], -double(2.0*i+1)/mesh.dx, 1e-8);
  }
}

TEST(MeshTest, solve) {
  Mesh mesh;

  // Zero charge density
  // d^2 u / dx^2 = 1
  // u = x(x-1)/2 = x^2/2 - x/2
  // to satisfy u(0) = u(1) = 0
  for(int i = 0; i < mesh.nmesh; i++){
  	  mesh.charge_density[i] = 0.0;
  	  std::cout << mesh.charge_density[i] << " ";
  }
  std::cout << "\n";

  mesh.solve_for_potential();

  double x;
  for(int i = 0; i < mesh.nmesh; i++){
	x = mesh.mesh[i];
	//std::cout << mesh.potential[i] << " " << 0.5*x*(x-1.0) << "\n";
	ASSERT_NEAR(mesh.potential[i], 0.5*x*(x-1.0), 1e-8);
  }

  // Sine charge density
  // d^2 u / dx^2 = 1 - sin(2*pi*x)
  // u = sin(2*pi*x)/(4 pi**2) + x(x-1)/2 = sin(2*pi*x)/(4 pi**2) + x^2/2 - x/2
  // to satisfy u(0) = u(1) = 0
  for(int i = 0; i < mesh.nmesh; i++){
  	  mesh.charge_density[i] = sin(2.0*M_PI*mesh.mesh[i]);
  	  std::cout << mesh.charge_density[i] << " ";
  }
  std::cout << "\n";

  mesh.solve_for_potential();

  for(int i = 0; i < mesh.nmesh; i++){
	x = mesh.mesh[i];
	//std::cout << mesh.potential[i] << " " << 0.5*x*(x-1.0) + sin(2.0*M_PI*x)/(4.0*M_PI*M_PI)<< "\n";
	ASSERT_NEAR(mesh.potential[i], 0.5*x*(x-1.0) + sin(2.0*M_PI*x)/(4.0*M_PI*M_PI), 1e-3);
  }

}

TEST(MeshTest, solve_for_potential_fft) {
  Mesh mesh;
  int N = mesh.nmesh;
  FFT fft(mesh.nintervals);

  // Poisson equation
  // d^2 u / dx^2 = 1 - charge_density
  //
  // Zero RHS
  // d^2 u / dx^2 = 0
  // charge_density = 1
  // u = 0
  for(int i = 0; i < N; i++){
  	  mesh.charge_density[i] = 1.0;
  	  //std::cout << mesh.charge_density[i] << " ";
  }
  //std::cout << "\n";

  mesh.solve_for_potential_fft(&fft);

  for(int i = 0; i < N; i++){
	ASSERT_NEAR(mesh.potential[i], 0.0, 1e-8);
  	//std::cout << mesh.potential[i] << " ";
  }
  //std::cout << "\n";

  // Poisson equation
  // d^2 u / dx^2 = 1 - charge_density
  //
  // charge_density = 1 - cos(k*x)
  // d^2 u / dx^2 = cos(k*x)
  // u = - cos(k*x)/k**2
  double x, k;
  k = mesh.k[1];
  for(int i = 0; i < N; i++){
	x = mesh.mesh[i];
  	mesh.charge_density[i] = 1.0 - cos(k*x);
  }
  mesh.solve_for_potential_fft(&fft);

  for(int i = 0; i < N; i++){
	x = mesh.mesh[i];
	ASSERT_NEAR(mesh.potential[i], -cos(k*x)/(k*k), 1e-8);
  }

  // Poisson equation
  // d^2 u / dx^2 = 1 - charge_density
  //
  // charge_density = 1 - sin(k*x)
  // d^2 u / dx^2 = sin(k*x)
  // u = - sin(k*x)/k**2
  int k_ind = 7;
  k = mesh.k[k_ind];
  for(int i = 0; i < N; i++){
	x = mesh.mesh[i];
  	mesh.charge_density[i] = 1.0 - sin(k*x);
  }
  mesh.solve_for_potential_fft(&fft);

  for(int i = 0; i < N; i++){
	x = mesh.mesh[i];
	ASSERT_NEAR(mesh.potential[i], -sin(k*x)/(k*k), 1e-8);
  }

}

TEST(MeshTest, solve_for_electric_field_fft) {
  Mesh mesh;
  int N = mesh.nmesh;
  FFT fft(mesh.nintervals);

  // Poisson equation
  // d^2 u / dx^2 = 1 - charge_density
  //
  // Zero RHS
  // d^2 u / dx^2 = 0
  // charge_density = 1
  // u = 0
  // E = - Grad(phi) = 0
  for(int i = 0; i < N; i++){
  	  mesh.charge_density[i] = 1.0;
  }

  mesh.solve_for_electric_field_fft(&fft);

  for(int i = 0; i < N; i++){
	ASSERT_NEAR(mesh.electric_field[i], 0.0, 1e-8);
  }

  // Poisson equation
  // d^2 u / dx^2 = 1 - charge_density
  //
  // charge_density = 1 - cos(k*x)
  // d^2 u / dx^2 = cos(k*x)
  // u = - cos(k*x)/k**2
  // E = - Grad(u) = - sin(k*x)/k
  double x, k;
  k = mesh.k[1];
  std::cout << k << "\n";
  for(int i = 0; i < N; i++){
	x = mesh.mesh[i];
  	mesh.charge_density[i] = 1.0 - cos(k*x);
  }
  mesh.solve_for_electric_field_fft(&fft);

  for(int i = 0; i < N; i++){
	x = mesh.mesh[i];
	ASSERT_NEAR(mesh.electric_field[i], -sin(k*x)/k, 1e-8);
  }

  // Poisson equation
  // d^2 u / dx^2 = 1 - charge_density
  //
  // charge_density = 1 - sin(k*x)
  // d^2 u / dx^2 = sin(k*x)
  // u = - sin(k*x)/k**2
  // E = - Grad(u) = cos(k*x)/k
  int k_ind = 7;
  k = mesh.k[k_ind];
  for(int i = 0; i < N; i++){
	x = mesh.mesh[i];
  	mesh.charge_density[i] = 1.0 - sin(k*x);
  }
  mesh.solve_for_electric_field_fft(&fft);

  for(int i = 0; i < N; i++){
	x = mesh.mesh[i];
	ASSERT_NEAR(mesh.electric_field[i], cos(k*x)/k, 1e-8);
  }

}

TEST(MeshTest, get_E_staggered_from_E) {
  Mesh mesh;
  for(int i = 0; i < mesh.nmesh; i++){
	mesh.electric_field[i] = double(i);
  }

  mesh.get_E_staggered_from_E();

  for(int i = 0; i < mesh.nmesh-1; i++){
  	EXPECT_EQ(mesh.electric_field_staggered[i], double(i+0.5));
  }
}
