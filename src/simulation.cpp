
#include "simulation.hpp"

/*
 * Initialize all code components
 */
//void initialize() {

  // TODO: [Set input parameters]

//};

/*
 * Evolve simulation through all timesteps
 */
void evolve(Mesh *mesh, Plasma *plasma, FFT *fft) {

  for (int i = 0; i < mesh->nt; i++) {
    plasma->push(mesh);
    mesh->deposit(plasma);
    mesh->solve_for_electric_field_fft(fft);
    // TODO: implement real diagnostics!
    for (int j = 0; j < mesh->nmesh-1; j++){
    	std::cout << mesh->electric_field[j] << " ";
    }
    std::cout << "\n";
  };
};
