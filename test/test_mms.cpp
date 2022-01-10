#include <gtest/gtest.h>
#include "mesh.hpp"
#include "plasma.hpp"
#include <cmath>
#include <boost/math/statistics/linear_regression.hpp>

TEST(MMSTest, SpatialInitialConditions) {
  Mesh mesh;
  Plasma plasma;

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
	Plasma plasma(np);

	mesh.deposit(&plasma);
//	for(int i = 0; i < mesh.nmesh; i++){
//	  std::cout << mesh.charge_density[i] << " ";
//	}
//	std::cout << "\n";

	error = 0.0;
	for(int i = 0; i < mesh.nmesh; i++){
	  error += std::abs(mesh.charge_density[i] - (1.0 + 0.01 * cos( 2.0*M_PI*mesh.mesh[i]))/double(mesh.nintervals));
	}

	log_nparticles.push_back(std::log(np));
	log_summed_error.push_back(std::log(error));
  }


//  for(int j = 0; j < 20; j++){
//	  std::cout << log_summed_error[j] << " ";
//  }

  using boost::math::statistics::simple_ordinary_least_squares;
  auto [c0, c1] = simple_ordinary_least_squares(log_nparticles, log_summed_error);
  //std::cout << "f(x) = " << c0 << " + " << c1 << "*x" << "\n";

  // summed error should decay like nparticles^{-1/2}
  ASSERT_NEAR(c1, -0.5, 0.05);
}

