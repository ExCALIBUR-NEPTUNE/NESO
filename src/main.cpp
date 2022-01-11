#include "main.hpp"
#include "plasma.hpp"
#include "mesh.hpp"
#include "diagnostics.hpp"
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
  Mesh mesh(100);
  Plasma plasma(10000);
  Diagnostics diagnostics;
  FFT fft(mesh.nintervals);
  evolve(&mesh,&plasma,&fft,&diagnostics);
  
  return 0;
};

/*
 * Initialize all code components
 */
//void initialize() {

  // TODO: [Set input parameters]

//};

/*
 * Evolve simulation through all timesteps
 */
void evolve(Mesh *mesh, Plasma *plasma, FFT *fft, Diagnostics *diagnostics) {

  for (int i = 0; i < mesh->nt; i++) {
    plasma->push(mesh);
    mesh->deposit(plasma);
    mesh->solve_for_electric_field_fft(fft);
    diagnostics->compute_total_energy(mesh,plasma);
    // TODO: implement real diagnostics!
//    for (int j = 0; j < mesh->nmesh-1; j++){
//    	std::cout << mesh->electric_field[j] << " ";
//    }
//    std::cout << "\n";
//    double t = double(i+1)*mesh->dt;
//    for (int j = 0; j < plasma->n; j++){
//    	std::cout << t << " " << plasma->x[j] << " " << plasma->v[j] << "\n";
//    }
  };
  for(int i = 0; i < mesh->nt; i++){
	  std::cout << double(i)*mesh->dt << " " << diagnostics->total_energy.at(i) << " " << diagnostics->particle_energy.at(i) << " " << diagnostics->field_energy.at(i) << "\n";
  }
};
