/*
 * Module for dealing with particles
 */

#include "species.hpp"
#include "mesh.hpp"
#include <string>
#include <iostream>
#include <cmath>
#include <random>

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

/*
 * Initialize particles
 */
Species::Species(bool kinetic_in, double T_in, double q_in, double m_in, int n_in) {

	// Whether this species is treated kinetically (true) or adiabatically (false)
	kinetic = kinetic_in;

	// These quantities are all normalized to the quantities for a
	// fictitious reference species

	// Species temperature
	T = T_in;
	// Species charge
	q = q_in;
	// Species mass
	m = m_in;
	// Species thermal velocity
	vth = std::sqrt(2*T/m);

	if( kinetic ){
		// Number of particles
		n = n_in;
		
		set_array_dimensions();
		set_initial_conditions(x, v);

		for(int i = 0; i < n; i++){
			w[i] = 1.0/double(n);
		}
	}
	// adiabatic species
	else {
		charge_density = q;
	}
}

/*
 * Set array dimensions for all the properties relating to the particles
 */
void Species::set_array_dimensions() {

	x.resize(n);   // particle x positions
	v.x.resize(n); // particle x velocities
	v.y.resize(n); // particle y velocities
	v.z.resize(n); // particle z velocities
	w.resize(n);   // particle weight

}

/*
 * Initialize distribution function
 * Pick random triplet (pos, vel, r) and keep particle if r < f(x,v)
 * for f the initial distribution.
 */
void Species::set_initial_conditions(std::vector<double> &x, Velocity &v) {

	// trial particle positions and velocities
	double pos, vel, r;
	// amplitude of wave perturbation
	double amp = 1e-8; 
	double big = 1e8;

	int i = 0;
	std::default_random_engine generator;
	while( i < n ){
		pos = std::uniform_real_distribution<double>(0.0,1)(generator);
		vel = std::uniform_real_distribution<double>(-6.0,6.0)(generator);
		r = std::uniform_real_distribution<double>(0.0,1.0)(generator);

		//if( r * (1.0 + amp) < (1.0 + amp * cos( 2.0*M_PI*pos)) * exp(-vel*vel) / sqrt(M_PI) ){
		//if( r * (1.0 + amp) < (1.0 + amp * cos( 2.0*M_PI*pos)) * exp(-vel*vel) ){
		//if( r  <  amp * cos( 2.0*M_PI*pos) * exp(-vel*vel) / sqrt(M_PI) ) {
		//if( r < exp(-vel*vel) ) {
		//if( r < 0.5 * big * ( exp(- big*(vel-0.5)*(vel-0.5)) + exp(- big*(vel+0.5)*(vel+0.5)) )) {
		if( r < 0.5 * big * ( exp(- big*(vel-1.0)*(vel-1.0)) + exp(- big*(vel+1.0)*(vel+1.0)) )) {
			x.at(i) = pos;
			v.x.at(i) = vel;
			v.y.at(i) = 0.0;
			v.z.at(i) = 0.0;
			i++;
		}
	}
}

/*
 * Second order accurate particle pusher
 * with spatially periodic boundary conditions
 */
void Species::push(Mesh *mesh) {

	for(int i = 0; i < n; i++) {
		v.x.at(i) += 0.5 * mesh->dt * mesh->evaluate_electric_field(x.at(i)) * q / (m * vth);
		x.at(i) += mesh->dt * v.x.at(i) * vth ;

              	//apply periodic bcs
               	while(x.at(i) < 0){
                       x.at(i) += 1.0;
               	}
               	x.at(i) = std::fmod(x.at(i), 1.0);

               	v.x.at(i) += 0.5 * mesh->dt * mesh->evaluate_electric_field(x.at(i)) * q / (m * vth);

       }
}

void Species::sycl_push(Mesh *mesh) {

  	size_t dataSize = n;
  	try {
    		auto asyncHandler = [&](sycl::exception_list exceptionList) {
      		for (auto& e : exceptionList) {
        		std::rethrow_exception(e);
      		}
    	};
    	auto defaultQueue = sycl::queue{sycl::default_selector{}, asyncHandler};

        const double dx_coef = mesh->dt * vth;
        const double dv_coef = 0.5 * mesh->dt * q / (m * vth);

    	sycl::buffer<double,1> vx_h(v.x.data(), sycl::range<1>{dataSize});
    	sycl::buffer<double,1> x_h(x.data(), sycl::range<1>{dataSize});
    	sycl::buffer<double,1> mesh_h(mesh->mesh.data(), sycl::range<1>{mesh->mesh.size()});
    	sycl::buffer<double,1> electric_field_h(mesh->electric_field.data(), sycl::range<1>{mesh->electric_field.size()});
    	auto dx_coef_h = sycl::buffer{&dx_coef, sycl::range{1}};
    	auto dv_coef_h = sycl::buffer{&dv_coef, sycl::range{1}};

    	defaultQueue
        	.submit([&](sycl::handler& cgh) {
          		auto vx_d = vx_h.get_access<sycl::access::mode::read_write>(cgh);
          		auto x_d = x_h.get_access<sycl::access::mode::read_write>(cgh);
          		auto electric_field_d = electric_field_h.get_access<sycl::access::mode::read_write>(cgh);
          		auto mesh_d = mesh_h.get_access<sycl::access::mode::read_write>(cgh);
          		auto dx_coef_d = dx_coef_h.get_access<sycl::access::mode::read>(cgh);
          		auto dv_coef_d = dv_coef_h.get_access<sycl::access::mode::read>(cgh);

          		cgh.parallel_for<>(
              			sycl::range{dataSize},
              			[=](sycl::id<1> idx) { 
					
					// First half-push v
	  				vx_d[idx] += dv_coef_d[0] * mesh->sycl_evaluate_electric_field(mesh_d, electric_field_d, x_d[idx]);

					// Push x
         				x_d[idx] += dx_coef_d[0] * vx_d[idx];
					while(x_d[idx] < 0){
						x_d[idx] += 1.0;
					}
                			x_d[idx] = std::fmod(x_d[idx], 1.0);

					// Second half-push v
         				vx_d[idx] += dv_coef_d[0] * mesh->sycl_evaluate_electric_field(mesh_d, electric_field_d, x_d[idx]);
				}
			);
        	})
        	.wait();

    	defaultQueue.throw_asynchronous();
  	} catch (const sycl::exception& e) {
    		std::cout << "Exception caught: " << e.what() << std::endl;
  	}
}
