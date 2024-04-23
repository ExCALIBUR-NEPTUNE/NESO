/*
 * Module for dealing with diagnostics
 */
#include "diagnostics.hpp"
#include <cmath>

/*
 * Store simulation time as a vector
 */
void Diagnostics::store_time(const double t) { time.push_back(t); }

/*
 * Compute and store total energy
 */
void Diagnostics::compute_total_energy(sycl::queue &Q, Mesh &mesh,
                                       Plasma &plasma) {

  compute_field_energy(Q, mesh);
  compute_particle_energy(Q, plasma);

  total_energy.push_back(field_energy.back() + particle_energy.back());
}

/*
 * Compute and store the energy in the electric field
 */
void Diagnostics::compute_field_energy(sycl::queue &Q, Mesh &mesh) {

  const int num_steps = mesh.electric_field.size() - 1;
  std::vector<double> data(num_steps);

  // Create buffer using host allocated "data" array
  sycl::buffer<double, 1> buf(data.data(), sycl::range<1>{data.size()});

  sycl::buffer<double, 1> electric_field_d(
      mesh.electric_field.data(), sycl::range<1>{mesh.electric_field.size()});
  Q.submit([&](sycl::handler &h) {
    auto writeresult = sycl::accessor(buf, h);
    auto electric_field_a = sycl::accessor(electric_field_d, h);
    h.parallel_for(sycl::range<1>{size_t(num_steps)}, [=](auto idx) {
      writeresult[idx[0]] = std::pow(electric_field_a[idx], 2);
    });
  });
  Q.wait();

  // Single task is needed here to make sure
  // data is not written over.
  Q.submit([&](sycl::handler &h) {
    sycl::accessor a(buf, h);
    h.single_task([=]() {
      for (int i = 1; i < num_steps; i++)
        a[0] += a[i];
    });
  });
  Q.wait();

  sycl::host_accessor answer(buf, sycl::read_only);
  double energy = answer[0] * 0.5 / std::pow(mesh.normalized_box_length, 2);

  field_energy.push_back(energy);
}

/*
 * Compute and store the energy in the particles
 */
void Diagnostics::compute_particle_energy(sycl::queue &Q, Plasma &plasma) {

  double energy = 0.0;
  for (std::size_t j = 0; j < plasma.n_kinetic_spec; j++) {
    // double energy_spec = 0.0;
    const int n = plasma.kinetic_species.at(j).n;
    double data[n];

    // Create buffer using host allocated "data" array
    sycl::buffer<double, 1> buf(data, sycl::range<1>{size_t(n)});
    sycl::buffer<double, 1> vx_d(
        plasma.kinetic_species.at(j).v.x.data(),
        sycl::range<1>{plasma.kinetic_species.at(j).v.x.size()});
    sycl::buffer<double, 1> vy_d(
        plasma.kinetic_species.at(j).v.y.data(),
        sycl::range<1>{plasma.kinetic_species.at(j).v.y.size()});
    sycl::buffer<double, 1> vz_d(
        plasma.kinetic_species.at(j).v.z.data(),
        sycl::range<1>{plasma.kinetic_species.at(j).v.z.size()});
    sycl::buffer<double, 1> w_d(
        plasma.kinetic_species.at(j).w.data(),
        sycl::range<1>{plasma.kinetic_species.at(j).w.size()});

    Q.submit([&](sycl::handler &h) {
      sycl::accessor species_energy(buf, h, sycl::write_only);
      auto vx_a = vx_d.get_access<sycl::access::mode::read_write>(h);
      auto vy_a = vy_d.get_access<sycl::access::mode::read_write>(h);
      auto vz_a = vz_d.get_access<sycl::access::mode::read_write>(h);
      auto w_a = w_d.get_access<sycl::access::mode::read_write>(h);

      h.parallel_for(sycl::range<1>{size_t(n)}, [=](sycl::id<1> idx) {
        species_energy[idx[0]] =
            w_a[idx] * (vx_a[idx] * vx_a[idx] + vy_a[idx] * vy_a[idx] +
                        vz_a[idx] * vz_a[idx]);
      });
    });
    Q.wait();

    // Single task is needed here to make sure
    // data is not written over.
    Q.submit([&](sycl::handler &h) {
      sycl::accessor a(buf, h);
      h.single_task([=]() {
        for (int i = 1; i < n; i++)
          a[0] += a[i];
      });
    });
    Q.wait();

    sycl::host_accessor answer(buf, sycl::read_only);
    double energy_spec = answer[0] * plasma.kinetic_species.at(j).m;
    energy += energy_spec;
  }
  energy *= 0.5;

  particle_energy.push_back(energy);
}
