#ifndef NEPTUNE_CUSTOM_TYPES_H
#define NEPTUNE_CUSTOM_TYPES_H

#include "sycl_typedefs.hpp"
#include <complex>
#include <vector>

// distribution function arrays
using Complex = std::complex<double>;
using Distribution = std::vector<Complex>;
using SYCLFlatDistribution = sycl::buffer<std::complex<double>>;
using Space = std::vector<std::complex<double>>;
using SpaceReal = std::vector<double>;

#endif // NEPTUNE_CUSTOM_TYPES_H
