#ifndef NEPTUNE_CUSTOM_TYPES_H
#define NEPTUNE_CUSTOM_TYPES_H

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <complex>

// distribution function arrays
using Complex = std::complex<double>;
using Distribution = std::vector<Complex>;
using SYCLFlatDistribution = sycl::buffer<std::complex<double>>;
using Space = std::vector<std::complex<double>>;
using SpaceReal = std::vector<double>;

#endif // NEPTUNE_CUSTOM_TYPES_H
