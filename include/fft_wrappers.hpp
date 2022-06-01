#ifndef NEPTUNE_FFTWRAPPERS_H
#define NEPTUNE_FFTWRAPPERS_H

#ifdef INTEL_MKL_FFT
    #include "fft_mkl.hpp"
#else
    #include "fft_fftw.hpp"
#endif


#endif
