#include <gtest/gtest.h>
#include "fft.hpp"
#include <cmath>
#include <random>

TEST(FFTTest, FFT) {
	FFT f(16);
	EXPECT_EQ(f.N, 16);
}

void initialize_zero(fftw_complex *x, const int N){

	for(int i = 0; i < N; i++) {
		x[i][0] = 0.0;
		x[i][1] = 0.0;
	}
}

TEST(FFTTest, ForwardSingleModes) {

	int N = 16; 
	FFT f(N);

	double *x, *k;
	x = new double[N];
	k = new double[N];
	for(int i = 0; i < N; i++){
		x[i] = double(i)/double(N);
		k[i] = 2.0*M_PI*double(i);
	}

	// Zeroth mode: transform is constant 1
	initialize_zero(f.in,f.N);
	f.in[0][0] = 1.0;
	fftw_execute(f.plan_forward);
	for(int i = 0; i < f.N; i++){
		ASSERT_NEAR(f.out[i][0], 1.0, 1e-8);
		ASSERT_NEAR(f.out[i][1], 0.0, 1e-8);
	}

	// First mode: transform is exp(-i k[1] x)
	int k_ind = 1;
	initialize_zero(f.in,f.N);
	f.in[k_ind][0] = 1.0;
	fftw_execute(f.plan_forward);
	for(int i = 0; i < f.N; i++){
		ASSERT_NEAR(f.out[i][0], cos(k[k_ind]*x[i]), 1e-8);
		ASSERT_NEAR(f.out[i][1], -sin(k[k_ind]*x[i]), 1e-8);
	}

	// Thirteenth mode: transform is exp(-i k[13] x)
 	k_ind = 13;
	initialize_zero(f.in,f.N);
	f.in[k_ind][0] = 1.0;
	fftw_execute(f.plan_forward);
	for(int i = 0; i < f.N; i++){
		ASSERT_NEAR(f.out[i][0], cos(k[k_ind]*x[i]), 1e-8);
		ASSERT_NEAR(f.out[i][1], -sin(k[k_ind]*x[i]), 1e-8);
	}
}

TEST(FFTTest, InverseSingleModes) {

	int N = 5; 
	FFT f(N);

	double *x, *k;
	x = new double[N];
	k = new double[N];
	for(int i = 0; i < N; i++){
		x[i] = double(i)/double(N);
		k[i] = 2.0*M_PI*double(i);
	}

	// Zeroth mode: transform is constant 1
	initialize_zero(f.out,f.N);
	f.out[0][0] = 1.0;
	fftw_execute(f.plan_inverse);
	for(int i = 0; i < f.N; i++){
		ASSERT_NEAR(f.in[i][0], 1.0, 1e-8);
		ASSERT_NEAR(f.in[i][1], 0.0, 1e-8);
	}

	// First mode: transform is exp(i k[1] x)
	int k_ind = 1;
	initialize_zero(f.out,f.N);
	f.out[k_ind][0] = 1.0;
	fftw_execute(f.plan_inverse);
	for(int i = 0; i < f.N; i++){
		ASSERT_NEAR(f.in[i][0], cos(k[k_ind]*x[i]), 1e-8);
		ASSERT_NEAR(f.in[i][1], sin(k[k_ind]*x[i]), 1e-8);
	}

	// Third mode: transform is exp(i k[3] x)
 	k_ind = 3;
	initialize_zero(f.out,f.N);
	f.out[k_ind][0] = 1.0;
	fftw_execute(f.plan_inverse);
	for(int i = 0; i < f.N; i++){
		ASSERT_NEAR(f.in[i][0], cos(k[k_ind]*x[i]), 1e-8);
		ASSERT_NEAR(f.in[i][1], sin(k[k_ind]*x[i]), 1e-8);
	}
}

/*
 * Forward followed by backward transforms should yield array * N
 */
TEST(FFTTest, ForwardInverse) {

	int N = 7; 
	FFT f(N);

	fftw_complex *result;
	result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);

	// Random input array
	std::default_random_engine generator;
	for(int i = 0; i < f.N; i++){
		result[i][0] = std::uniform_real_distribution<double>(-1.0,1.0)(generator);
		f.in[i][0] = result[i][0];
		result[i][1] = std::uniform_real_distribution<double>(-1.0,1.0)(generator);
		f.in[i][1] = result[i][1];
	}

	// Perform forward transform
	fftw_execute(f.plan_forward);

	// Zero input array, to make sure
	// that we are using output of the
	// inverse transform
	initialize_zero(f.in,f.N);

	// Perform inverse transform
	fftw_execute(f.plan_inverse);

	for(int i = 0; i < f.N; i++){
		ASSERT_NEAR(f.in[i][0]/double(f.N), result[i][0], 1e-8);
		ASSERT_NEAR(f.in[i][1]/double(f.N), result[i][1], 1e-8);
	}
}

