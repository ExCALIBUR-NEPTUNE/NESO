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
void evolve(Mesh *mesh, Plasma *plasma, FFT *fft, Diagnostics *diagnostics) {

  for (int i = 0; i < mesh->nt; i++) {

    mesh->t += mesh->dt;
    diagnostics->store_time(mesh->t);

    plasma->push(mesh);
    mesh->deposit(plasma);
    mesh->solve_for_electric_field_fft(fft);
    diagnostics->compute_total_energy(mesh,plasma);
  };
  for(int i = 0; i < mesh->nt; i++){
    std::cout << double(i)*mesh->dt << " " << diagnostics->total_energy.at(i) << " " << diagnostics->particle_energy.at(i) << " " << diagnostics->field_energy.at(i) << "\n";
  }
  for(int i = 0; i < mesh->nt; i++){
	  //std::cout << double(i)*mesh->dt << " " << diagnostics->total_energy.at(i) << " " << diagnostics->particle_energy.at(i) << " " << diagnostics->field_energy.at(i) << "\n";
	  std::cout << diagnostics->total_energy.at(i) << " " << diagnostics->particle_energy.at(i) << " " << diagnostics->field_energy.at(i) << "\n";
  }
};
