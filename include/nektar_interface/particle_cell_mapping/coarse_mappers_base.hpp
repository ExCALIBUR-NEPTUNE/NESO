#ifndef __COARSE_MAPPERS_BASE_H_
#define __COARSE_MAPPERS_BASE_H_

#include "coarse_lookup_map.hpp"
#include <memory>

namespace NESO {

class CoarseMappersBase {
protected:
  /// SYCLTarget on which to perform computations.
  SYCLTargetSharedPtr sycl_target;
  /// The coarse lookup map used to find geometry objects.
  std::unique_ptr<CoarseLookupMap> coarse_lookup_map;
  /// The nektar++ cell id for the cells indices pointed to from the map.
  std::unique_ptr<BufferDeviceHost<int>> dh_cell_ids;
  /// The MPI rank that owns the cell.
  std::unique_ptr<BufferDeviceHost<int>> dh_mpi_ranks;
  /// The type of the cell, e.g. a quad or a triangle.
  std::unique_ptr<BufferDeviceHost<int>> dh_type;
  /// Instance to throw kernel errors with.
  std::unique_ptr<ErrorPropagate> ep;

public:
  CoarseMappersBase(SYCLTargetSharedPtr sycl_target)
      : sycl_target(sycl_target),
        ep(std::make_unique<ErrorPropagate>(sycl_target)){};
};

} // namespace NESO

#endif
