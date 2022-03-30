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
Species::Species(const Mesh &mesh, bool kinetic_in, double T_in, double q_in, double m_in, int n_in) : dx_coef_d(1), dv_coef_d(1), x_d(1), vx_d(1), vy_d(1), vz_d(1), w_d(1), charge_density_d(1), kinetic(kinetic_in), T(T_in), q(q_in), m(m_in), vth(std::sqrt(2*T/m)), n(n_in) {

	if( kinetic ){
		set_array_dimensions();
		set_initial_conditions(x, v);
		x_d = sycl::buffer<double,1>(x);
		x_d.set_write_back(false);
    		vx_d = sycl::buffer<double,1>(v.x);
		vx_d.set_write_back(false);
    		vy_d = sycl::buffer<double,1>(v.y.data(), sycl::range<1>{v.y.size()});
    		vz_d = sycl::buffer<double,1>(v.z.data(), sycl::range<1>{v.z.size()});

		for(int i = 0; i < n; i++){
			w[i] = 1.0/double(n);
		}
    		w_d = sycl::buffer<double,1>(w.data(), sycl::range<1>{w.size()});
	}
	// adiabatic species
	else {
		charge_density = q;
    		charge_density_d = sycl::buffer{&charge_density, sycl::range{1}};
	}

	// Need to remove const if using adaptive timestepping
        dx_coef = mesh.dt * vth;
        dv_coef = 0.5 * mesh.dt * q / (m * vth);

    	dx_coef_d = sycl::buffer{&dx_coef, sycl::range{1}};
    	dx_coef_d.set_write_back(false);
    	dv_coef_d = sycl::buffer{&dv_coef, sycl::range{1}};
    	dv_coef_d.set_write_back(false);

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
void Species::push(Mesh &mesh) {

	for(int i = 0; i < n; i++) {
         	v.x.at(i) += 0.5 * mesh.dt * mesh.evaluate_electric_field(x.at(i)) * q / (m * vth);
         	x.at(i) += mesh.dt * v.x.at(i) * vth ;

              	//apply periodic bcs
               	while(x.at(i) < 0){
                       x.at(i) += 1.0;
               	}
               	x.at(i) = std::fmod(x.at(i), 1.0);

         	v.x.at(i) += 0.5 * mesh.dt * mesh.evaluate_electric_field(x.at(i)) * q / (m * vth);

       }
}

void Species::sycl_push(sycl::queue &queue, Mesh *mesh) {

    	queue.submit([&](sycl::handler& cgh) {
          		auto vx_a = vx_d.get_access<sycl::access::mode::read_write>(cgh);
          		auto x_a = x_d.get_access<sycl::access::mode::read_write>(cgh);
          		auto electric_field_a = mesh->electric_field_d.get_access<sycl::access::mode::read_write>(cgh);
          		auto mesh_a = mesh->mesh_d.get_access<sycl::access::mode::read_write>(cgh);
          		auto dx_coef_a = dx_coef_d.get_access<sycl::access::mode::read>(cgh);
          		auto dv_coef_a = dv_coef_d.get_access<sycl::access::mode::read>(cgh);

          		cgh.parallel_for<>(
              			sycl::range{size_t(n)},
              			[=](sycl::id<1> idx) { 
					
					// First half-push v
	  				vx_a[idx] += dv_coef_a[0] * mesh->sycl_evaluate_electric_field(mesh_a, electric_field_a, x_a[idx]);

					// Push x
         				x_a[idx] += dx_coef_a[0] * vx_a[idx];
					while(x_a[idx] < 0){
						x_a[idx] += 1.0;
					}
                			x_a[idx] = std::fmod(x_a[idx], 1.0);

					// Second half-push v
         				vx_a[idx] += dv_coef_a[0] * mesh->sycl_evaluate_electric_field(mesh_a, electric_field_a, x_a[idx]);
				}
			);
        	})
        	.wait();

    	queue.throw_asynchronous();
}
