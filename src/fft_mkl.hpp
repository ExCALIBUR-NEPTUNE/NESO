#ifndef NEPTUNE_FFTMKL_H
#define NEPTUNE_FFTMKL_H

#include <CL/sycl.hpp>
#include "oneapi/mkl/dfti.hpp"
#include "custom_types.hpp"

class FFT {
public:

	FFT(sycl::queue &q_in, int N_in) : Q(q_in), N(N_in), N_d(N_in), plan(N_in), init_plan(false) {}
	//~FFT();

	// Size of FFT
	int N;
	sycl::buffer<size_t,1> N_d;

  	oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,oneapi::mkl::dft::domain::COMPLEX> plan;
	
	// Out-of-place c2c forward transform on sycl buffer
	void forward(sycl::buffer<Complex,1> &in_d, sycl::buffer<Complex,1> &out_d){
		if (not init_plan) make_plan(plan);
		oneapi::mkl::dft::compute_forward(plan,in_d,out_d);
	}

	// Out-of-place c2c backward transform on sycl buffer
	void backward(sycl::buffer<Complex,1> &in_d, sycl::buffer<Complex,1> &out_d){
		if (not init_plan) make_plan(plan);
		oneapi::mkl::dft::compute_backward(plan,in_d,out_d);
	}

private:
	template<typename onemkl_plan_type>
	void make_plan(onemkl_plan_type &plan) const{
		plan.commit(Q);
		init_plan = true;
		Q.wait();
	}

	sycl::queue &Q;

	mutable bool init_plan;

};

#endif // NEPTUNE_FFT_MKL_H
