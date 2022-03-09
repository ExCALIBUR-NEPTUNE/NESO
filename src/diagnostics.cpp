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
void Diagnostics::compute_total_energy(Mesh &mesh, Plasma &plasma){

	compute_field_energy(mesh);
	compute_particle_energy(plasma);

	total_energy.push_back(field_energy.back() + particle_energy.back());

}

/*
 * Compute and store the energy in the electric field
 */
void Diagnostics::compute_field_energy(Mesh &mesh) {

	const int num_steps = mesh.electric_field.size()-1;
	sycl::queue myQueue{sycl::property::queue::in_order()};
  auto policy = dpl::execution::make_device_policy(
      sycl::queue(sycl::default_selector{}, dpc_common::exception_handler));

	//double energy = 0.0;
//	// NB: Omit periodic point
//	for( std::size_t i = 0; i < mesh.electric_field.size()-1; i++) {
//		energy += std::pow(mesh.electric_field.at(i),2);
//	}
//	energy *= 0.5 / std::pow(mesh.normalized_box_length,2);

//  auto electric_field_a = mesh.electric_field_d.get_access<sycl::access::mode::read_write>();
//
//  double energy = std::transform_reduce(
//      policy, dpl::counting_iterator<int>(1),
//      oneapi::dpl::counting_iterator<int>(n), 0.0f, std::plus<double>(),
//      [=](int i) {
//        //float x = (float)(((float)i - 0.5f) / (float(num_steps)));
//	//double x = std::pow(mesh.electric_field.at(i),2);
//	double x = std::pow(electric_field_a[i],2);
//        return x;
//      });

  double data[num_steps];

  // Create buffer using host allocated "data" array
  sycl::buffer<double, 1> buf{data, sycl::range<1>{size_t(num_steps)}};

  policy.queue().submit([&](sycl::handler& h) {
		  sycl::accessor writeresult(buf,h,sycl::write_only);
      auto electric_field_a = mesh.electric_field_d.get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>{size_t(num_steps)}, [=](sycl::id<1> idx) {
      // float x = ((float)idx[0] - 0.5) / (float)num_steps;
      double x = std::pow(electric_field_a[idx],2);
      writeresult[idx[0]] = x;
    });
  });
  policy.queue().wait();

  // Single task is needed here to make sure
  // data is not written over.
  policy.queue().submit([&](sycl::handler& h) {
		  sycl::accessor a(buf,h);
    h.single_task([=]() {
      for (int i = 1; i < num_steps; i++) a[0] += a[i];
    });
  });
  policy.queue().wait();


  // float mynewresult = buf.get_access<access::mode::read>()[0] / (float)num_steps;
  sycl::host_accessor answer(buf,sycl::read_only) ; 
  double mynewresult = answer[0]; 

  double energy = mynewresult * 0.5 / std::pow(mesh.normalized_box_length,2);

  field_energy.push_back(energy);
}

/*
 * Compute and store the energy in the particles
 */
void Diagnostics::compute_particle_energy(const Plasma &plasma) {

	double energy = 0.0;
	for( std::size_t j = 0; j < plasma.n_kinetic_spec; j++) {
		for( std::size_t i = 0; i < plasma.kinetic_species.at(j).n; i++) {
			energy += plasma.kinetic_species.at(j).w.at(i)*
				  plasma.kinetic_species.at(j).m*
				(
					std::pow(plasma.kinetic_species.at(j).v.x.at(i),2)
					+ std::pow(plasma.kinetic_species.at(j).v.y.at(i),2)
					+ std::pow(plasma.kinetic_species.at(j).v.z.at(i),2)
				);
		}
	}
	energy *= 0.5;

	particle_energy.push_back(energy);
}
