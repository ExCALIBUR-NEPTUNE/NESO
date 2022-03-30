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
void evolve(sycl::queue &q, Mesh &mesh, Plasma &plasma, FFT &fft, Diagnostics &diagnostics) {

  for (int i = 0; i < mesh.nt; i++) {

    mesh.t += mesh.dt;
    diagnostics.store_time(mesh.t);

    plasma.push(q, mesh);
    //mesh.deposit(plasma);
    mesh.sycl_deposit(q, plasma);
    //mesh.solve_for_electric_field_fft(q, fft);
    mesh.sycl_solve_for_electric_field_fft(q, fft);
    diagnostics.compute_total_energy(mesh,plasma);
    // TODO: implement real diagnostics!
    //for (int j = 0; j < mesh->nmesh-1; j++){
    //	std::cout << mesh->electric_field[j] << " ";
    //}
    //std::cout << "\n";
//    double t = double(i+1)*mesh->dt;
//    for (int j = 0; j < plasma->n; j++){
//    	std::cout << t << " " << plasma->x[j] << " " << plasma->v[j] << "\n";
//    }
  };
  for(int i = 0; i < mesh.nt; i++){
	  //std::cout << double(i)*mesh->dt << " " << diagnostics->total_energy.at(i) << " " << diagnostics->particle_energy.at(i) << " " << diagnostics->field_energy.at(i) << "\n";
	  std::cout << diagnostics.total_energy.at(i) << " " << diagnostics.particle_energy.at(i) << " " << diagnostics.field_energy.at(i) << "\n";
  }
};
