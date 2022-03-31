/*
 * Module for dealing with diagnostics
 */

#include "plasma.hpp"
#include "mesh.hpp"
#include "diagnostics.hpp"
//#include <string>
//#include <iostream>
#include <cmath>

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include "dpc_common.hpp"


/*
 * Store simulation time as a vector
 */
void Diagnostics::store_time(const double t){
	time.push_back(t);
}

/*
 * Compute and store total energy
 */
void Diagnostics::compute_total_energy(sycl::queue &Q, Mesh &mesh, Plasma &plasma){

	compute_field_energy(Q,mesh);
	compute_particle_energy(Q,plasma);

	total_energy.push_back(field_energy.back() + particle_energy.back());

}

/*
 * Compute and store the energy in the electric field
 */
void Diagnostics::compute_field_energy(sycl::queue &Q, Mesh &mesh) {

	const int num_steps = mesh.electric_field.size()-1;
	std::vector<double> data(num_steps);

  	// Create buffer using host allocated "data" array
  	sycl::buffer buf(data);

  	Q.submit([&](sycl::handler& h) {
		auto writeresult = sycl::accessor(buf,h);
      		auto electric_field_a = sycl::accessor(mesh.electric_field_d,h);
    		h.parallel_for(size_t(num_steps), [=](auto idx) {
      			writeresult[idx[0]] = std::pow(electric_field_a[idx],2);
    		});
  	});
  	Q.wait();

  	// Single task is needed here to make sure
  	// data is not written over.
  	Q.submit([&](sycl::handler& h) {
		sycl::accessor a(buf,h);
    		h.single_task([=]() {
      			for (int i = 1; i < num_steps; i++) a[0] += a[i];
    		});
  	});
  	Q.wait();

  	sycl::host_accessor answer(buf,sycl::read_only) ; 
  	double energy = answer[0] * 0.5 / std::pow(mesh.normalized_box_length,2);

  	field_energy.push_back(energy);
}

/*
 * Compute and store the energy in the particles
 */
void Diagnostics::compute_particle_energy(sycl::queue &Q, Plasma &plasma) {

	double energy = 0.0;
	for( std::size_t j = 0; j < plasma.n_kinetic_spec; j++) {
		//double energy_spec = 0.0;
		const int n = plasma.kinetic_species.at(j).n;
		double data[n];

  		// Create buffer using host allocated "data" array
  		sycl::buffer<double, 1> buf{data, sycl::range<1>{size_t(n)}};

  		Q.submit([&](sycl::handler& h) {
			sycl::accessor species_energy(buf,h,sycl::write_only);
      			auto vx_a = plasma.kinetic_species.at(j).vx_d.get_access<sycl::access::mode::read_write>(h);
      			auto vy_a = plasma.kinetic_species.at(j).vy_d.get_access<sycl::access::mode::read_write>(h);
      			auto vz_a = plasma.kinetic_species.at(j).vz_d.get_access<sycl::access::mode::read_write>(h);
      			auto w_a = plasma.kinetic_species.at(j).w_d.get_access<sycl::access::mode::read_write>(h);

    			h.parallel_for(sycl::range<1>{size_t(n)}, [=](sycl::id<1> idx) {
      				species_energy[idx[0]] = w_a[idx] * 
							( vx_a[idx]*vx_a[idx]
							  + vy_a[idx]*vy_a[idx]
							  + vz_a[idx]*vz_a[idx] );
    			});
  		});
  		Q.wait();

  		// Single task is needed here to make sure
  		// data is not written over.
  		Q.submit([&](sycl::handler& h) {
			sycl::accessor a(buf,h);
    			h.single_task([=]() {
      				for (int i = 1; i < n; i++) a[0] += a[i];
    			});
  		});
  		Q.wait();

  		sycl::host_accessor answer(buf,sycl::read_only) ; 
  		double energy_spec = answer[0] * plasma.kinetic_species.at(j).m ;
		energy += energy_spec;
	}
	energy *= 0.5;

	particle_energy.push_back(energy);
}
