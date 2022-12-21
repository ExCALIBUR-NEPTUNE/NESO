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

constexpr double INVERSE_GOLDEN_RATIOS[10] = {
    0.6180339887498948, 0.7548776662466927, 0.8191725133961644,
    0.8566748838545029, 0.8812714616335696, 0.8986537126286992,
    0.9115923534820549, 0.9215993196339829, 0.9295701282320229,
    0.9360691110777584};

/**
 *  Return the ith number in the additive reccurence "r-sequence"
 * of uniform distributed quasi-random numbers.
 *
 *  @param i The ith number in the sequence
 *  @param irrational The irrational number upon which the sequence is based
 *  @param seed The offset number of the sequence
 *  @returns a double
 */
inline double rsequence(const int i, const double irrational,
                        const double seed = 0.0) {
  return std::fmod(i * irrational + seed, 1);
}

/**
 *  Return the ith number in the additive reccurence "r-sequence"
 * of uniform distributed quasi-random numbers.
 *
 *  @param i The ith number in the sequence
 *  @param dim Choose the inverse of the dimth hyper golden ratio
 *  as the irrational number to base the sequence on
 *  @param seed The offset number of the sequence
 *  @returns a double
 */
inline double rsequence(const int i, const int dim, const double seed = 0.0) {
  return rsequence(i, INVERSE_GOLDEN_RATIOS[dim], seed);
}

/**
 *  Create a uniform distribution of particle positions within a set of extents
 * via an R-sequence quasi-random number generator, using the inverse of
 * hyper golden ratios.
 *
 *  @param N Number of points to generate.
 *  @param ndim Number of dimensions.
 *  @param extents Extent of each of the dimensions.
 *  @param seed Seed to sequence
 *  @returns (N)x(ndim) set of positions stored for each column.
 */
inline std::vector<std::vector<double>>
rsequence_within_extents(const int N, const int ndim, const double *extents,
                         const double seed = 0.0) {

  std::vector<std::vector<double>> positions(ndim);
  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(N);
  }

  for (int dimx = 0; dimx < ndim; dimx++) {
    auto &positions_dimx = positions[dimx];
    const auto igr = INVERSE_GOLDEN_RATIOS[dimx];
    const double ex = extents[dimx];
    for (int px = 0; px < N; px++) {
      const double samplex = rsequence(px, igr, seed) * ex;
      positions_dimx[px] = samplex;
    }
  }

  return positions;
}

} // namespace NESO

#endif
