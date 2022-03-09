#include "main.hpp"
#include "plasma.hpp"
#include "mesh.hpp"
#include "diagnostics.hpp"
#include "simulation.hpp"
#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <string>
#include <iostream>

class hello_world;

int main() {
  try {
	auto asyncHandler = [&](sycl::exception_list exceptionList) {
	for (auto& e : exceptionList) {
		std::rethrow_exception(e);
	}
  };
  auto q = sycl::queue{sycl::default_selector{}, asyncHandler};

  //initialize();
  // Initialize by calling Mesh and Particle constructors
  Mesh mesh(128,0.05,40);
  Species ions(mesh,false,2.0,-1,1836.2,1);
  Species electrons(mesh,true,2.0,1,1,12800);
  std::vector<Species> species_list;
  species_list.push_back(ions);
  species_list.push_back(electrons);
  Plasma plasma(species_list);

  Diagnostics diagnostics;
  FFT fft(mesh.nintervals);

  mesh.set_initial_field(mesh,plasma,fft);
  evolve(q,mesh,plasma,fft,diagnostics);
  } catch (const sycl::exception& e) {
   		std::cout << "Exception caught: " << e.what() << std::endl;
 }
  
  return 0;
};
