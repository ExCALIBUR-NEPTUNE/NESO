/*
 * Module for performing 1D FFTs with FFTW3
 */

#include "fft.hpp"
#include <fftw3.h>

/*
 * Initial class
 */
FFT::FFT(int N_in) {

	bool initialized = false;

	// Size of FFT array
	N = N_in;

	// Real space array
	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
	// Transformed space array
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);

	plan_forward = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	plan_inverse = fftw_plan_dft_1d(N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

}

void FFT::initialize_FFT(){

	// Return if already initialized
	if(initialized){
		return;
	}



	initialized = true;

}
