#include <gtest/gtest.h>
#include "../src/fft_mkl.hpp"
#include <cmath>
#include <complex>
#include <random>

TEST(FFTMKLTest, FFT) {
	FFT_MKL f(16);
	EXPECT_EQ(f.N, 16);
}

void initialize_zero(sycl::queue &q, sycl::buffer<double,1> &x_buf, const int Nk){

	q.submit([&](sycl::handler &h) {
		sycl::accessor x_acc{x_buf, h, sycl::write_only};
          	h.parallel_for<>( 
			sycl::range{size_t(Nk)},
              		[=](sycl::id<1> idx) { 
        			x_acc[idx] = 0.0;
      			});
	}).wait();
}

void set_mode_k(sycl::queue &q, sycl::buffer<double,1> &x_buf, const int k_ind, const double value){

    	q.submit([&](sycl::handler &h) {
		sycl::accessor x_acc{x_buf, h, sycl::write_only};
      		h.single_task<>([=]() {
        		x_acc[k_ind] = value;
      		});
    	}).wait();
}

TEST(FFTMKLTest, ForwardSingleModes) {

	auto asyncHandler = [&](sycl::exception_list exceptionList) {};
	auto q = sycl::queue{sycl::default_selector{}, asyncHandler};

	int N = 32; 
	FFT_MKL f(N);

	double *x, *k;
	x = new double[N];
	k = new double[N];
	for(int i = 0; i < N; i++){
		x[i] = double(i)/double(N);
		k[i] = 2.0*M_PI*double(i);
	}

    	oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL> transform_plan(f.N);
    	transform_plan.commit(q);

	for(int k_ind = 0; k_ind < f.N; k_ind++){

		initialize_zero(q,f.in_d,f.N);
		set_mode_k(q,f.in_d,k_ind,1.0);

		// NB: This uses a local
		// definition of in_d (not the one
		// from fft_mkl.hpp. Defining it in the scope below forces an update of the buffer f.out when out_d goes out of scope.
		{
		sycl::buffer<double,1> out_d(f.out.data(), sycl::range<1>{f.out.size()});
		oneapi::mkl::dft::compute_forward(transform_plan, f.in_d, out_d);
		}

		for(int i = 0; i < (f.N/2); i++){
			//std::cout << f.out[2*i] << " " << f.out[2*i+1] << " " << cos(k[k_ind]*x[i]) << " " << -sin(k[k_ind]*x[i]) <<  "\n";
			ASSERT_NEAR(f.out[2*i], cos(k[k_ind]*x[i]), 1e-8);
			ASSERT_NEAR(f.out[2*i+1], -sin(k[k_ind]*x[i]), 1e-8);
		}
	}
}

//TEST(FFTMKLTest, BackwardSingleModes) {
//
//	auto asyncHandler = [&](sycl::exception_list exceptionList) {};
//	auto q = sycl::queue{sycl::default_selector{}, asyncHandler};
//
//	int N = 8; 
//	FFT_MKL f(N);
//
//	double *x, *k;
//	x = new double[N];
//	k = new double[N];
//	for(int i = 0; i < N; i++){
//		x[i] = double(i)/double(N);
//		k[i] = 2.0*M_PI*double(i);
//	}
//
//    	oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL> transform_plan(f.N);
//    	transform_plan.commit(q);
//
//	for(int k_ind = 0; k_ind < (f.N/2); k_ind++){
//
//		//std::cout << k_ind << "\n";
//		initialize_zero(q,f.out_d,(f.N/2));
//		set_mode_k(q,f.out_d,k_ind,1.0);
//
//		// NB: This uses a local
//		// definition of in_d (not the one
//		// from fft_mkl.hpp. Defining it in the scope below forces an update of the buffer f.out when out_d goes out of scope.
//		{
//		sycl::buffer<double,1> in_d(f.in); //.data(), sycl::range<1>{f.out.size()});
//		oneapi::mkl::dft::compute_backward(transform_plan, f.out_d, in_d);
//		}
//
//		for(int i = 0; i < f.N; i++){
//			std::cout << f.in[i] << " " << cos(k[k_ind]*x[i]) << " " << sin(k[k_ind]*x[i]) << "\n";
//			//ASSERT_NEAR(f.in[2*i], cos(k[k_ind]*x[i]), 1e-8);
//			//ASSERT_NEAR(f.in[2*i+1], sin(k[k_ind]*x[i]), 1e-8);
//		}
//	}
//}

/*
 * Forward followed by backward transforms should yield array * N
 */
TEST(FFTMKLTest, ForwardInverse) {

	auto asyncHandler = [&](sycl::exception_list exceptionList) {};
	auto q = sycl::queue{sycl::default_selector{}, asyncHandler};

	int N = 16; 
	FFT_MKL f(N);

	std::vector<double> result(N);

	// Random input array
	std::default_random_engine generator;
	for(int i = 0; i < f.N; i++){
		result[i] = std::uniform_real_distribution<double>(-1.0,1.0)(generator);
	}

    	oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL> transform_plan(N);
    	transform_plan.commit(q);

	{
	sycl::buffer<double,1> in_d(result.data(), sycl::range<1>{result.size()});
	sycl::buffer<double,1> fwdbwd_d(f.in.data(), sycl::range<1>{f.in.size()});

	initialize_zero(q,fwdbwd_d,f.N);

    	oneapi::mkl::dft::compute_forward(transform_plan, in_d, f.out_d);
	//initialize_zero(q,in_d,f.N);
    	oneapi::mkl::dft::compute_backward(transform_plan, f.out_d, fwdbwd_d);
	}

	for(int i = 0; i < f.N; i++){
		ASSERT_NEAR(f.in[i]/double(f.N), result[i], 1e-8);
	}
}

//	int N = 16; 
//	FFT_MKL f(N);
//
//	fftw_complex *result;
//	result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
//
//	// Random input array
//	std::default_random_engine generator;
//	for(int i = 0; i < f.N; i++){
//		result[i][0] = std::uniform_real_distribution<double>(-1.0,1.0)(generator);
//		f.in[i][0] = result[i][0];
//		result[i][1] = std::uniform_real_distribution<double>(-1.0,1.0)(generator);
//		f.in[i][1] = result[i][1];
//	}
//
//	// Perform forward transform
//	fftw_execute(f.plan_forward);
//
//	// Zero input array, to make sure
//	// that we are using output of the
//	// inverse transform
//	initialize_zero(f.in,f.N);
//
//	// Perform inverse transform
//	fftw_execute(f.plan_inverse);
//
//	for(int i = 0; i < f.N; i++){
//		ASSERT_NEAR(f.in[i][0]/double(f.N), result[i][0], 1e-8);
//		ASSERT_NEAR(f.in[i][1]/double(f.N), result[i][1], 1e-8);

