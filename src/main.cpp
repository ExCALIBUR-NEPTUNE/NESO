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
  auto defaultQueue = sycl::queue{};

  defaultQueue
      .submit([&](sycl::handler& cgh) {
        auto os = sycl::stream{128, 128, cgh};

        //cgh.single_task<hello_world>([=]() { os << "Hello World!\n"; });
      })
      .wait();

  //initialize();
  // Initialize by calling Mesh and Particle constructors
  Mesh mesh(32);
  Species ions(1,1.0,1,false);
  Species electrons(32,1,-1,true);
  std::vector<Species> species_list;
  species_list.push_back(ions);
  species_list.push_back(electrons);
  Plasma plasma(species_list);

  Diagnostics diagnostics;
  FFT fft(mesh.nintervals);

  mesh.set_initial_field(&mesh,&plasma,&fft);
  evolve(&mesh,&plasma,&fft,&diagnostics);
  
  return 0;
};
