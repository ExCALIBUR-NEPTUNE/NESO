class FFT;

#ifndef __FFT_H__
#define __FFT_H__

#include <fftw3.h>

class FFT {
public:

	FFT(int N);

	// Size of FFT
	int N;

	// Real space array
	fftw_complex *in;

	// Transformed space array
	fftw_complex *out;

	// real to complex plan
	fftw_plan plan_forward;

	// complex to real plan
	fftw_plan plan_inverse;


};

#endif // __FFT_H__
