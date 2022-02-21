#include <gtest/gtest.h>
#include "../src/mesh.hpp"
#include "../src/plasma.hpp"
#include "../src/diagnostics.hpp"
#include "../src/simulation.hpp"
#include <cmath>
#include <boost/math/statistics/linear_regression.hpp>

TEST(MMSTest, SpatialInitialConditions) {
  Mesh mesh;
  Species electrons(true);
  std::vector<Species> species_list;
  species_list.push_back(electrons);
  Plasma plasma(species_list);

  std::vector<double> log_nparticles;
  std::vector<double> log_summed_error;
  double error;

  // Deposit particles onto mesh, and test that the resulting
  // distribution tends to a cosine.
  int np = 1;

  // j < 20, 3.6 secs
  // j < 11, 0.01 secs
  for(int j = 0; j < 11; j++){
  	np *= 2;
	Species electrons(true,1.0,1,1,np);
	species_list.at(0) = electrons;
  	Plasma plasma(species_list);

	mesh.deposit(&plasma);

	error = 0.0;
	for(int i = 0; i < mesh.nmesh; i++){
	  error += std::abs(mesh.charge_density[i] - (1.0 + 0.01 * cos( 2.0*M_PI*mesh.mesh[i]))/double(mesh.nintervals));
	}

	log_nparticles.push_back(std::log(np));
	log_summed_error.push_back(std::log(error));
  }

  using boost::math::statistics::simple_ordinary_least_squares;
  auto [c0, c1] = simple_ordinary_least_squares(log_nparticles, log_summed_error);
  //std::cout << "f(x) = " << c0 << " + " << c1 << "*x" << "\n";

  // summed error should decay like nparticles^{-1/2}
  ASSERT_NEAR(c1, -0.5, 0.05);
}

/*
 * Test the growth rate of the two stream
 * instability
 */
TEST(MMSTest, TwoStreamGrowthRate) {

  // Unconverged
  //Mesh mesh(32,0.05,80);
  //Plasma plasma(3200);

  // The number of timesteps is chosen so
  // that the run finishes as the
  // instability saturates.
  Mesh mesh(64,0.05,40);
  Species electrons(true,1.0,-1,1,6400);
  Species ions(false,1.0,1836,1);
  std::vector<Species> species_list;
  species_list.push_back(electrons);
  species_list.push_back(ions);
  Plasma plasma(species_list);

  // Also work, but take longer:
  //Mesh mesh(128,0.05,40);
  //Plasma plasma(12800);
  //Mesh mesh(256,0.05,25);
  //Plasma plasma(25600);

  Diagnostics diagnostics;
  FFT fft(mesh.nintervals);

  mesh.set_initial_field(&mesh,&plasma,&fft);
  evolve(&mesh,&plasma,&fft,&diagnostics);

  std::vector<double> log_field_energy;
  for(int j = 0; j < diagnostics.field_energy.size(); j++){
	log_field_energy.push_back(std::log(diagnostics.field_energy.at(j)));
  }

  // Put a line of best fit through field_energy = cst * electric_field^2
  // The electric field has the theoretical
  // normalized growth rate of sqrt(15)/2
  // Thus field_energy should have a growth
  // rate of twice this, sqrt(15).

  using boost::math::statistics::simple_ordinary_least_squares;
  auto [c0, c1] = simple_ordinary_least_squares(diagnostics.time, log_field_energy);
  //std::cout << "f(x) = " << c0 << " + " << c1 << "*x" << "\n";

  // Instability should have grow rate
  // sqrt(15), but the fitting process is
  // quite rough
  const double sqrt15 = std::sqrt(15.0);
  ASSERT_NEAR((c1-sqrt15)/sqrt15,0.0, 0.1);
}

