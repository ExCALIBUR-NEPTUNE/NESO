/*
 * Module for dealing with particles
 */

#include "mesh.hpp"
#include "custom_types.hpp"
#include "fft_mkl.hpp"
#include "oneapi/mkl/dfti.hpp"
#include "species.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

/*
 * Initialize mesh
 */
Mesh::Mesh(int nintervals_in, double dt_in, int nt_in)
    : nt(nt_in), nintervals(nintervals_in), nmesh(nintervals_in + 1), t(0.0),
      dt(dt_in), mesh_d(1), electric_field_d(1), charge_density_d(0), dx_d(0),
      poisson_E_factor_d(0), nmesh_d(0) {

  // size of grid spaces on a domain of length 1
  dx = 1.0 / double(nintervals);
  dx_d = sycl::buffer<double, 1>(&dx, sycl::range<1>{1});
  dx_d.set_write_back(false);

  nmesh_d = sycl::buffer<int, 1>(&nmesh, sycl::range<1>{1});
  nmesh_d.set_write_back(false);

  // box length in units of Debye length
  // NB default of nx makes each cell one Debye length
  normalized_box_length = double(nintervals);

  // mesh point vector
  mesh.resize(nmesh);

  for (std::size_t i = 0; i < mesh.size(); i++) {
    mesh.at(i) = double(i) * dx;
  }

  // constexpr size_t dataSize = nmesh;
  size_t dataSize = nmesh;

  std::vector<double> ints;
  ints.resize(dataSize);
  for (int i = 0; i < dataSize; ++i) {
    ints.at(i) = static_cast<double>(i);
  }

  sycl::buffer<double, 1> ints_h(ints.data(), sycl::range<1>{ints.size()});
  mesh_d = sycl::buffer<double, 1>(mesh.data(), sycl::range<1>{mesh.size()});
  mesh_d.set_write_back(false);

  //    	q.submit([&](sycl::handler& cgh) {
  //          		auto ints_d =
  //          ints_h.get_access<sycl::access::mode::read>(cgh); 		auto dx_d
  //          = dx_h.get_access<sycl::access::mode::read>(cgh);
  //          auto mesh_d = mesh_h.get_access<sycl::access::mode::write>(cgh);
  //
  //          		cgh.parallel_for<>(
  //              			sycl::range{dataSize},
  //              			[=](sycl::id<1> idx) {
  //					mesh_d[idx] = ints_d[idx] * dx_d[0];
  //				}
  //			);
  //        	})
  //        	.wait();
  //
  //    	q.throw_asynchronous();

  // mesh point vector staggered at half points
  mesh_staggered.resize(nintervals);
  for (std::size_t i = 0; i < mesh_staggered.size(); i++) {
    mesh_staggered.at(i) = double(i + 0.5) * dx;
  }
  // Fourier wavenumbers k[j] =
  // 2*pi*j such that k*mesh gives
  // the argument of the exponential
  // in FFTW
  // NB: for complex to complex transforms, the second half of Fourier
  // modes must be negative of the first half
  k.resize(nmesh);
  for (std::size_t i = 0; i < (k.size() / 2) + 1; i++) {
    k.at(i) = 2.0 * M_PI * double(i);
    k.at(k.size() - i - 1) = -k.at(i);
  }
  // Poisson factor
  // Coefficient to multiply Fourier-transformed charge density by in the
  // Poisson solve:
  // - L**2 / (lambda_D**2 * k**2 * nmesh)
  // This accounts for the change of sign, the wavenumber squared (for
  // the Laplacian), and the length of the array (the normalization in
  // the FFT).
  poisson_factor.resize(nintervals);
  poisson_factor.at(0) = 0.0;
  for (std::size_t i = 1; i < poisson_factor.size(); i++) {
    poisson_factor.at(i) = -std::pow(normalized_box_length, 2) /
                           (k.at(i) * k.at(i) * double(nintervals));
  }

  // Poisson Electric field factor
  // Coefficient to multiply Fourier-transformed charge density by in the
  // Poisson solve:
  // i * L^2 / ( lambda_D^2 * k * nmesh)
  // to obtain the electric field from the Poisson equation and
  // E = - Grad(phi) in a single step.
  // This accounts for the change of sign, the wavenumber squared (for
  // the Laplacian), the length of the array (the normalization in the
  // FFT), and the factor of (-ik) for taking the Grad in fourier space.
  // NB Rather than deal with complex arithmetic, we make this a real
  // number that we apply to the relevant array entry.
  poisson_E_factor.resize(nintervals);
  poisson_E_factor.at(0).real(0.0);
  poisson_E_factor.at(0).imag(0.0);
  for (std::size_t i = 1; i < poisson_E_factor.size(); i++) {
    poisson_E_factor.at(i).real(0.0);
    poisson_E_factor.at(i).imag(std::pow(normalized_box_length, 2) /
                                (k.at(i) * double(nintervals)));
  }
  poisson_E_factor_d = sycl::buffer<Complex, 1>(
      poisson_E_factor.data(), sycl::range<1>{poisson_E_factor.size()});
  poisson_E_factor_d.set_write_back(false);

  charge_density.resize(nmesh);
  for (std::size_t i = 0; i < charge_density.size(); i++) {
    charge_density.at(i) = 0.0;
  }

  charge_density_d = sycl::buffer<double, 1>(
      charge_density.data(), sycl::range<1>{charge_density.size()});
  charge_density_d.set_write_back(false);

  // Electric field on mesh
  electric_field.resize(nmesh);
  for (std::size_t i = 0; i < electric_field.size(); i++) {
    electric_field.at(i) = 0.0;
  }

  electric_field_d = sycl::buffer<double, 1>(
      electric_field.data(), sycl::range<1>{electric_field.size()});
  electric_field_d.set_write_back(false);

  // Electric field on staggered mesh
  electric_field_staggered.resize(nintervals);
  for (std::size_t i = 0; i < electric_field_staggered.size(); i++) {
    electric_field_staggered.at(i) = 0.0;
  }
  potential.resize(nmesh);

  // NB these must be double * for use in lapack call
  // super diagonal
  du = new double[nmesh - 1];
  // sub diagonal
  dl = new double[nmesh - 1];
  // diagonal
  d = new double[nmesh];
  // right hand side: - dx^2 * charge density
  b = new double[nmesh];
}

/*
// Invert real double tridiagonal matrix with lapack
extern "C" {
        //int dgttrs_(char trans, int n, int nrhs, const double *dl, const
double *d, const double *du, const double *du2, const int *ipiv, double *b, int
ldb); void dgtsv_(int *n, int *nrhs, double *dl, double *d, double *du, double
*b, int *ldb, int *info);
}
*/

/*
 * Given a point and a mesh, return the index of the grid point immediately to
 * the left of x. When we include both endpoints on the mesh, the point x is
 * always between the grid's upper and lower bounds, so the we only need the
 * left-hand index
 */
int Mesh::get_left_index(const double x, const std::vector<double> mesh) {

  int index = 0;
  while (mesh.at(index + 1) < x and index < int(mesh.size())) {
    index++;
  };
  return index;
}

int Mesh::sycl_get_left_index(const double x,
                              const sycl::accessor<double> mesh_d) {

  int index = 0;
  while (mesh_d[index + 1] < x and index < int(mesh.size())) {
    index++;
  };
  return index;
}

/*
 * Evaluate the electric field at x grid points by
 * interpolating onto the grid
 * SYCL note: this needs to be thread safe so it can be called inside the
 * pusher.
 */
double Mesh::evaluate_electric_field(const double x) {

  // Implementation of
  //   np.interp(x,self.half_grid,self.efield)
  //
  // Find grid cell that x is in
  int index = get_left_index(x, mesh);

  // std::cout << "index : " << index << " nmesh " << nmesh << "\n";
  // std::cout << index_down << " " << index_up  << "\n";
  // std::cout << mesh_staggered[index_down] << " " << x << " " <<
  // mesh_staggered[index_up]  << "\n";
  //  now x is in the cell ( mesh[index-1], mesh[index] )

  double cell_width = mesh.at(index + 1) - mesh.at(index);
  double distance_into_cell = x - mesh.at(index);

  // r is the proportion if the distance into the cell that the particle is at
  // e.g. midpoint => r = 0.5
  double r = distance_into_cell / cell_width;
  // std::cout << r  << "\n";
  return (1.0 - r) * electric_field.at(index) +
         r * electric_field.at(index + 1);
};

/*
 * Evaluate the electric field at x grid points by
 * interpolating onto the grid
 * SYCL note: this is a copy of evaluate_electric_field, but able to be called
 * in sycl. This should become evaluate_electric_field eventually
 */
double
Mesh::sycl_evaluate_electric_field(sycl::accessor<double> mesh_d,
                                   sycl::accessor<double> electric_field_d,
                                   double x) {

  // Find grid cell that x is in
  int index = sycl_get_left_index(x, mesh_d);

  // now x is in the cell ( mesh[index-1], mesh[index] )

  double cell_width = mesh_d[index + 1] - mesh_d[index];
  double distance_into_cell = x - mesh_d[index];

  // r is the proportion if the distance into the cell that the particle is at
  // e.g. midpoint => r = 0.5
  double r = distance_into_cell / cell_width;

  return (1.0 - r) * electric_field_d[index] + r * electric_field_d[index + 1];
};

/*
 * Deposit charge at grid points. For a particle in a grid space [a,b],
 * distribute charge at grid points a and b proportionally to the particle's
 * distance from those points.
 */
void Mesh::deposit(Plasma &plasma) {

  // Zero the density before depositing
  for (std::size_t i = 0; i < charge_density.size(); i++) {
    charge_density.at(i) = 0.0;
  }

  // Deposit particles
  for (int j = 0; j < plasma.n_kinetic_spec; j++) {
    for (int i = 0; i < plasma.kinetic_species.at(j).n; i++) {
      // get index of left-hand grid point
      int index = (floor(plasma.kinetic_species.at(j).x.at(i) / dx));
      // r is the proportion if the distance into the cell that the particle is
      // at e.g. midpoint => r = 0.5
      double r = plasma.kinetic_species.at(j).x.at(i) / dx - double(index);

      // if a particle is on the furthest right grid point,
      // move it to the periodic point
      // but we must do this after calculating r
      if (index == nintervals) {
        index = 0;
      }

      charge_density.at(index) += (1.0 - r) *
                                  plasma.kinetic_species.at(j).w.at(i) *
                                  plasma.kinetic_species.at(j).q;
      charge_density.at(index + 1) += r * plasma.kinetic_species.at(j).w.at(i) *
                                      plasma.kinetic_species.at(j).q;
    }
  }

  // Add charge from adiabatic species
  for (int j = 0; j < plasma.n_adiabatic_spec; j++) {
    for (int i = 0; i < nintervals; i++) {
      charge_density.at(i) += plasma.adiabatic_species.at(j).charge_density;
    }
  }

  // Ensure result is periodic.
  // The charge index 0 should have contributions from [0,dx] and [1-dx,1],
  // but at this point will only have the [0,dx] contribution. All the
  // [1-dx,1] contribution is at index nmesh-1. To make is periodic we
  // therefore sum the charges at the end points:
  charge_density.at(0) += charge_density.at(charge_density.size() - 1);
  // Then make the far boundary equal the near boundary
  charge_density.at(charge_density.size() - 1) = charge_density.at(0);

  //	for( int i = 0; i < nmesh; i++){
  //		std::cout << charge_density.at(i) << "\n";
  //	}
}

void Mesh::sycl_deposit(sycl::queue &Q, Plasma &plasma) {

  size_t nmesh = charge_density.size();
  size_t nthreads;
  // size_t wgsize;

  auto dev = Q.get_device();
  if (dev.is_cpu()) {
    nthreads = dev.get_info<sycl::info::device::max_compute_units>();
    // wgsize = dev.get_info<sycl::info::device::native_vector_width_double>() *
    // 2;
  } else {
    nthreads = dev.get_info<sycl::info::device::max_compute_units>(); // * 4;
    // wgsize = dev.get_info<sycl::info::device::max_work_group_size>();
  }

  sycl::buffer<double, 1> cd_long_d(sycl::range<1>{nthreads * nmesh},
                                    sycl::no_init);
  // Zero the density before depositing
  Q.submit([&](sycl::handler &cgh) {
     auto charge_density_a =
         charge_density_d.get_access<sycl::access::mode::write>(cgh);
     cgh.parallel_for(sycl::range{nmesh},
                      [=](sycl::id<1> idx) { charge_density_a[idx] = 0.0; });
   }).wait();

  // Deposit particles
  for (int j = 0; j < plasma.n_kinetic_spec; j++) {
    size_t nparticles = plasma.kinetic_species.at(j).n;
    auto q_d = sycl::buffer{&plasma.kinetic_species.at(j).q, sycl::range{1}};
    q_d.set_write_back(false);

    Q.submit([&](sycl::handler &cgh) {
       auto cd_long_a = cd_long_d.get_access<sycl::access::mode::write>(cgh);
       cgh.parallel_for(sycl::range{nmesh * nthreads},
                        [=](sycl::id<1> idx) { cd_long_a[idx] = 0.0; });
     }).wait();

    Q.submit([&](sycl::handler &cgh) {
       auto x_a = plasma.kinetic_species.at(j)
                      .x_d.get_access<sycl::access::mode::read>(cgh);
       auto w_a = plasma.kinetic_species.at(j)
                      .w_d.get_access<sycl::access::mode::read>(cgh);
       auto dx_a = dx_d.get_access<sycl::access::mode::read>(cgh);
       auto q_a = q_d.get_access<sycl::access::mode::read>(cgh);
       auto cd_long_a =
           cd_long_d.get_access<sycl::access::mode::read_write>(cgh);
       // sycl::stream out(65536, 256, cgh);

       cgh.parallel_for(sycl::range{nthreads}, [=](sycl::id<1> tid) {
         for (int idx = tid; idx < nparticles; idx += nthreads) {
           double position_ratio = x_a[idx] / dx_a[0];
           // get index of left-hand grid point
           int index = (floor(position_ratio));
           // r is the proportion if the distance into the cell that the
           // particle is at e.g. midpoint => r = 0.5
           double r = position_ratio - double(index);
           // out << "idx, tid, tid*nmesh + index, index = " << idx << " " <<
           // tid << " " << tid*nmesh + index << " " << index <<  sycl::endl;
           //  Update this thread's copy of charge_density
           cd_long_a[tid * nmesh + index] += (1.0 - r) * w_a[idx] * q_a[0];
           cd_long_a[tid * nmesh + index + 1] += r * w_a[idx] * q_a[0];
         }
       });
     }).wait();

    // Now reduce the copies of charge_density onto a single array
    Q.submit([&](sycl::handler &cgh) {
       auto charge_density_a =
           charge_density_d.get_access<sycl::access::mode::read_write>(cgh);
       auto cd_long_a = cd_long_d.get_access<sycl::access::mode::read>(cgh);
       // sycl::stream out(65536, 256, cgh);

       cgh.parallel_for(sycl::range{nmesh}, [=](sycl::id<1> idx) {
         for (int it = idx; it < nmesh * nthreads; it += nmesh) {
           charge_density_a[idx] += cd_long_a[it];
         }
       });
     }).wait();
  }

  // Add charge from adiabatic species
  Q.submit([&](sycl::handler &cgh) {
     auto charge_density_a =
         charge_density_d.get_access<sycl::access::mode::read_write>(cgh);
     for (int j = 0; j < plasma.n_adiabatic_spec; j++) {
       auto adiabatic_charge_density_a =
           plasma.adiabatic_species.at(j)
               .charge_density_d.get_access<sycl::access::mode::read>(cgh);

       cgh.parallel_for(sycl::range{size_t(nintervals)}, [=](sycl::id<1> idx) {
         charge_density_a[idx] += adiabatic_charge_density_a[0];
       });
     }
   }).wait();

  // Ensure result is periodic.
  // The charge index 0 should have contributions from [0,dx] and [1-dx,1],
  // but at this point will only have the [0,dx] contribution. All the
  // [1-dx,1] contribution is at index nmesh-1. To make is periodic we
  // therefore sum the charges at the end points:
  // charge_density.at(0) += charge_density.at(charge_density.size()-1);
  // Then make the far boundary equal the near boundary
  // charge_density.at(charge_density.size()-1) = charge_density.at(0);

  Q.submit([&](sycl::handler &cgh) {
     auto charge_density_a = sycl::accessor(charge_density_d, cgh);
     auto nmesh_a = sycl::accessor(nmesh_d, cgh);
     cgh.single_task([=]() {
       charge_density_a[0] += charge_density_a[nmesh - 1];
       charge_density_a[nmesh - 1] = charge_density_a[0];
     });
   }).wait();

  //	for( int i = 0; i < nmesh; i++){
  //		std::cout << charge_density.at(i) << "\n";
  //	}
}

/*
 * Solve Gauss' law for the electric field using the charge
 * distribution as the RHS. Combine this solve with definition
 * E = - Grad(phi) to do this in a single step.
 */
// void Mesh::solve_for_electric_field_fft(FFT &f) {
//
//	// Transform charge density (summed over species)
//	for(int i = 0; i < nintervals; i++) {
//         	f.in[i][0] = - charge_density.at(i);
//         	f.in[i][1] = 0.0;
//	}
//
//	fftw_execute(f.plan_forward);
//
//	// Divide by wavenumber
//	double tmp; // Working double to allow swap
//	for( std::size_t i = 0; i < poisson_E_factor.size(); i++) {
//		// New element = i * poisson_E_factor * old element
//		tmp = f.out[i][1];
//         	f.out[i][1] = poisson_E_factor.at(i) * f.out[i][0];
//		// Minus to account for factor of i
//         	f.out[i][0] = - poisson_E_factor.at(i) * tmp;
//	}
//
//	fftw_execute(f.plan_inverse);
//
//	for( std::size_t i = 0; i < electric_field.size()-1; i++) {
//		electric_field.at(i) = f.in[i][0];
//	}
//	electric_field.at(electric_field.size()-1) = electric_field.at(0);
//
// }

void Mesh::sycl_solve_for_electric_field_fft(sycl::queue &Q, FFT &f) {

  sycl::buffer<Complex, 1> transformed_charge_density_d(
      sycl::range<1>(size_t(f.N)), sycl::no_init);
  sycl::buffer<Complex, 1> in_d(sycl::range<1>(size_t(f.N)), sycl::no_init);

  // Transform charge density (summed over species)
  Q.submit([&](sycl::handler &cgh) {
     auto in_a = in_d.get_access<sycl::access::mode::write>(cgh);
     auto charge_density_a =
         charge_density_d.get_access<sycl::access::mode::read>(cgh);
     cgh.parallel_for<>(sycl::range{size_t(f.N)}, [=](sycl::id<1> idx) {
       in_a[idx] = -charge_density_a[idx];
     });
   }).wait();

  f.forward(in_d, transformed_charge_density_d);

  sycl::buffer<Complex, 1> sol_d(sycl::range<1>(size_t(f.N)), sycl::no_init);

  Q.submit([&](sycl::handler &cgh) {
     auto tcd_a =
         transformed_charge_density_d.get_access<sycl::access::mode::read>(cgh);
     auto sol_a = sol_d.get_access<sycl::access::mode::write>(cgh);
     auto poisson_E_factor_a =
         poisson_E_factor_d.get_access<sycl::access::mode::read>(cgh);
     cgh.parallel_for<>(sycl::range{size_t(f.N)}, [=](sycl::id<1> idx) {
       sol_a[idx] = poisson_E_factor_a[idx] * tcd_a[idx];
     });
   }).wait();

  sycl::buffer<Complex, 1> e_non_periodic_d(sycl::range<1>(size_t(f.N)),
                                            sycl::no_init);

  f.backward(sol_d, e_non_periodic_d);

  Q.submit([&](sycl::handler &cgh) {
    auto enp_a = e_non_periodic_d.get_access<sycl::access::mode::read>(cgh);
    auto electric_field_a =
        electric_field_d.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<>(sycl::range{size_t(f.N)}, [=](sycl::id<1> idx) {
      electric_field_a[idx] = enp_a[idx].real();
    });
  }); //.wait(); // no overlapping data access

  Q.submit([&](sycl::handler &cgh) {
     auto enp_a = e_non_periodic_d.get_access<sycl::access::mode::read>(cgh);
     auto electric_field_a =
         electric_field_d.get_access<sycl::access::mode::write>(cgh);
     auto N_a = nmesh_d.get_access<sycl::access::mode::read>(cgh);
     cgh.single_task([=]() { electric_field_a[N_a[0] - 1] = enp_a[0].real(); });
   }).wait();
}

/*
 * Solve Gauss' law for the electrostatic potential using the charge
 * distribution as the RHS. Take the FFT to diagonalize the problem.
 */
// void Mesh::solve_for_potential_fft(FFT &f) {
//
//	// Transform charge density (summed over species)
//	for(int i = 0; i < nintervals; i++) {
//         	f.in[i][0] = 1.0 - charge_density[i];
//         	f.in[i][1] = 0.0;
//	}
//
////	for(int i = 0; i < nmesh; i++) {
////		std::cout << f.in[i] << " ";
////	}
////	std::cout << "\n";
//	fftw_execute(f.plan_forward);
//
////	for(int i = 0; i < nintervals; i++) {
////		std::cout << f.out[i][0] << " " << f.out[i][1] << "\n";
////	}
////	std::cout << "\n";
//
//	// Divide by wavenumber
//	for(int i = 0; i < nintervals; i++) {
//        	f.out[i][0] *= poisson_factor[i];
//        	f.out[i][1] *= poisson_factor[i];
//	}
//	fftw_execute(f.plan_inverse);
//
//	for(int i = 0; i < nintervals; i++) {
//		potential[i] = f.in[i][0];
//		//std::cout << potential[i] << " ";
//	}
//	potential[nmesh-1] = potential[0];
//	//std::cout << "\n";
//
//}

/*
 * Solve Gauss' law for the electrostatic potential using the charge
 * distribution as a solve. In 1D, this is a tridiagonal matrix inversion with
 * the Thomas algorithm.
 */
/*
void Mesh::solve_for_potential() {

        // Initialize with general terms
        for(int i = 0; i < nmesh - 1; i++) {
                du[i] = 1.0;
                dl[i] = 1.0;
                d[i] = -2.0;
                b[i] = - dx * dx * (charge_density[i] - 1.0 );
        }
        d[nmesh-1] = -2.0;
        b[nmesh-1] = - dx * dx * ( charge_density[nmesh-1] - 1.0 );

        // apply boundary conditions
        // for simplicity use zero boundary conditions -
        // this is inconsistent with the periodic
        // particle pusher but allows us to use simple
        // tridiagonal inversion
        d[0] = 1.0;
        du[0] = 0.0;
        b[0] = 0.0;

        d[nmesh-1] = 1.0;
        dl[nmesh-2] = 0.0; // highest element, dl is offset by 1
        b[nmesh-1] = 0.0;

        int info;
        int nrhs = 1;
        int ldb = nmesh;
        dgtsv_(&nmesh, &nrhs, dl, d, du, b, &ldb, &info);

        // Could save a memcopy here by writing input RHS
        // to potential at top of function
        for(int i = 0; i < nmesh; i++) {
                potential[i] = b[i];
        }
}
*/

/*
 * Find the electric field by taking the gradient of the potential
 */
void Mesh::get_electric_field() {
  for (int i = 0; i < nmesh - 1; i++) {
    electric_field_staggered[i] = -(potential[i + 1] - potential[i]) / dx;
  }
}
/*
 * Interpolate electric field onto the staggered grid from the unstaggered grid
 */
void Mesh::get_E_staggered_from_E() {
  // E_staggered[i] is halfway between E[i] and E[i+1]
  // NB: No need for nmesh-1 index, as periodic point not kept in
  // electtic_field_staggered
  for (int i = 0; i < nmesh - 1; i++) {
    electric_field_staggered[i] =
        0.5 * (electric_field[i] + electric_field[i + 1]);
  }
}

/*
 * Set the electric field consistent with
 * the initial particle distribution
 */
void Mesh::set_initial_field(sycl::queue &Q, Mesh &mesh, Plasma &plasma,
                             FFT &fft) {
  // mesh.deposit(plasma);
  mesh.sycl_deposit(Q, plasma);
  // mesh.solve_for_electric_field_fft(fft);
  mesh.sycl_solve_for_electric_field_fft(Q, fft);
  // TODO: implement real diagnostics!
  //  for (int j = 0; j < mesh->nmesh-1; j++){
  //  	std::cout << mesh->electric_field[j] << " ";
  //  }
  //  std::cout << "\n";
}
