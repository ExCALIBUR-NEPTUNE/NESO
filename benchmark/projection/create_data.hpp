#pragma once
#include <sycl/sycl.hpp>
#include <gtest/gtest.h>
#include <nektar_interface/projection/device_data.hpp>
#include <utility>
#include <vector>
#include <random>

#define SEED 32423 

static inline void
random_fill_array(double *d_ptr, int size, sycl::queue &q)
{
	std::mt19937 random(SEED * SEED);
	std::uniform_real_distribution<double> dist(-20.0,20.0);
	std::vector<double> vec(0.0,size);
	std::generate(vec.begin(), vec.end(), [&]() { return dist{random};});
	q.copy<double>(d_ptr,vec.data(),size).wait_and_throw();
}

static inline 
std::vector<int>
dense_dist(int ncell, int min_per_cell, int max_per_cell)
{
    std::mt19937 random(SEED); // device());
    std::uniform_int_distribution<> pdist(min_per_cell, max_per_cell);
    std::uniform_int_distribution<> cdist(0, ncell);
    std::vector<int> npar(ncell, 0);

    std::for_each(npar.begin(), npar.end(), [&](int &n) { n = pdist(random); });
    return npar;
}

static inline std::vector<int>
simplesol_dist(int ncell, int active_cells, int min_per_cell, int max_per_cell)
{
    if (ncell == active_cells)
        return dense_dist(ncell, max_per_cell, min_per_cell);
    if (ncell < active_cells) {
        fprintf(stderr, "Active cells > total cells\n");
        exit(1);
    }
    // std::random_device device;
    assert(min_per_cell > 0);
    assert(max_per_cell >= min_per_cell);
    std::mt19937 random(2983); // device());
    std::uniform_int_distribution<> pdist(min_per_cell, max_per_cell);
    std::uniform_int_distribution<> cdist(0, ncell);
    std::vector<int> npar(ncell, 0);

    // anticipating ncell >> active_cells so this should be ok
    while (active_cells > 0) {
        int cell = cdist(random);
        if (npar[cell] == 0) {
            active_cells--;
            npar[cell] = pdist(random);
        }
    }
    return npar;
}


template<int nmode, typename Shape>
static inline auto create_data(sycl::queue &Q, 
							   int ndim,
							   int ncell,
							   int min_per_cell,
							   int max_per_cell,
							   int active_cells = -1)
{
  //Shove all the pointers in a vector so we can free them later

  std::vector<void *> all_pointers;

  //Par per cell
  int *par_per_cell = sycl::malloc_device<int>(ncell, Q);
  assert(par_per_cell);
  all_pointers.push_back((void *)par_per_cell);
  int max_row = 0;
  if (active_cell < 0 || active_cell == ncell) {
  	auto host_par_per_cell = dense_dist(ncell,min_per_cell,max_per_cell);
	max_row = *std::max_element(host_par_per_cell.begin(),host_par_per_cell.end());
	Q.copy<int>(par_per_cell,host_par_per_cell.data(),npar).wait();
  } else {
  	auto host_par_per_cell = simplesol_dist(ncell,active_cells,min_per_cell,max_per_cell);
	max_row = *std::max_element(host_par_per_cell.begin(),host_par_per_cell.end());
	Q.copy<int>(par_per_cell,host_par_per_cell.data(),npar).wait();
  }
  //DOFS 
  double *dofs = sycl::malloc_device<double>(npar * Shape::get_ndof<nmode>(), Q);
  assert(dofs);
  all_pointers.push_back((void *)dofs);
  Q.fill(dofs, double{val}, N).wait_and_throw();

  //DOF offsets ndof per cell but scanned
  auto host_offsets = std::vector<int>(Shape::get_ndof<nmode>(),ncell);
  std::exclusive_scan(host_offsets.begin(), host_offsets.end(),host_offsets.begin(),0);
  int *dof_offsets = sycl::malloc_device<int>(ncell, Q);
  assert(dof_offsets);
  all_pointers.push_back((void *)dof_offsets);
  Q.copy<int>(dof_offsets, host_offsets.data(), ncell).wait_and_throw();
  
  //Cell ids
  int *cell_ids = sycl::malloc_device<int>(ncell, Q);
  assert(cell_ids);
  all_pointers.push_back((void *)cell_ids);
  Q.parallel_for<>(ncell,[=] (sycl::id<1> id) {
		cell_ids[id] = id;		
  }).wait();

  //Particle postion pointers
  auto host_data_ptrs = std::vector<double*>(nullptr,ncell);
  for(int i = 0; i < ncell; i++) {	
		//allocate data arrays for positions
		double* par_data[3] = {nullptr};
 		par_data[0] = sycl::malloc_device<double>(host_par_per_cell[i],Q);
		assert(par_data[0]);
		random_fill_array(par_data[0],host_par_per_cell[i],Q);
	    all_pointers.push_back((void*)par_data[0]);
		if (ndim > 1) {
			par_data[1] = sycl::malloc_device<double>(host_par_per_cell[i],Q);
			assert(par_data[1]);
			random_fill_array(par_data[1],host_par_per_cell[i],Q);
			all_pointers.push_back((void*)par_data[1]);
		}
		if (ndim > 2) {
			par_data[2] = sycl::malloc_device<double>(host_par_per_cell[i],Q);
			random_fill_array(par_data[2],host_par_per_cell[i],Q);
			assert(par_data[2]);
			all_pointers.push_back((void*)par_data[2]);
		}
        //allocate array for cells
		host_data_ptrs[i] = sycl::malloc_device<double*>(ndim,Q);
		assert(host_data_ptrs[i]);
		all_pointers.push_back((void*)host_data_ptrs[i]);	
		Q.copy<double*>(host_data_ptrs[i],par_data,ndim);
  }
  auto positions = sycl::malloc_device<double**>(ncell,Q);
  assert(positions);
  all_pointers.push_back((void *)positions);
  Q.copy<double**>(positions,host_data_ptrs.data(),ncell); 
   
 
  //Particle value data
  
  auto host_value_ptrs = std::vector<double*>(nullptr,ncell);
  for(int i = 0; i < ncell; i++) {	
		//allocate value arrays for positions
 		auto par_value = sycl::malloc_device<double>(host_par_per_cell[i],Q);
		assert(par_value);
		random_fill_array(par_value,host_par_per_cell[i],Q);
	    all_pointers.push_back((void*)par_value);
		host_value_ptrs[i] = sycl::malloc_device<double*>(1,Q);
		assert(host_value_ptrs[i]);
		all_pointers.push_back((void*)host_value_ptrs[i]);	
		Q.copy<double*>(host_value_ptrs[i],par_value,1);
  }
  auto input = sycl::malloc_device<double**>(ncell,Q);
  assert(input);
  all_pointers.push_back((void *)input);
  Q.copy<double**>(input,host_value_ptrs.data(),ncell); 
  

  
  return std::pair(NESO::Project::DeviceData<double>(dofs, dof_offsets, ncell,max_row,
                                                     cell_ids, par_per_cell,
                                                     positions, input),
                   all_pointers);
}

static inline void free_data(sycl::queue &Q, std::vector<void *> &data) {
  for (auto &p : data) {
    sycl::free(p, Q);
  }
}
