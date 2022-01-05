#include "plasma.hpp"
#include "mesh.hpp"
#include "simulation.hpp"

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

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
  Mesh mesh(10);
  Plasma plasma;
  FFT fft(mesh.nintervals);
  evolve(&mesh,&plasma,&fft);
  
  return 0;
};

