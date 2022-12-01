#ifndef _NESO_UTILITY_POSITION_DIST_H_
#define _NESO_UTILITY_POSITION_DIST_H_

#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_real_distribution.hpp>

namespace NESO {

/**
 *  Create a uniform distribution of particle positions within a set of extents
 * via a Sobol quasi-random number generator.
 *
 *  @param N Number of points to generate.
 *  @param ndim Number of dimensions.
 *  @param extents Extent of each of the dimensions.
 *  @param discard Optional number of particles skip over before drawing
 * samples.
 *  @param seed Optional seed to use.
 *  @returns (N)x(ndim) set of positions stored for each column.
 */
inline std::vector<std::vector<double>>
sobol_within_extents(const int N, const int ndim, const double *extents,
                     uint64_t discard = 0, unsigned int seed = 0) {

  auto se = boost::random::sobol(ndim);
  se.seed(seed);
  se.discard(discard * ((uint64_t)ndim));
  auto uniform = boost::random::uniform_real_distribution(0.0, 1.0);

  std::vector<std::vector<double>> positions(ndim);
  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(N);
  }

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const double ex = extents[dimx];
      const double samplex = uniform(se) * ex;
      positions[dimx][px] = samplex;
    }
  }

  return positions;
}

} // namespace NESO

#endif
