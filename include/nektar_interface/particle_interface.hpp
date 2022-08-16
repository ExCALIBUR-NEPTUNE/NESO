#ifndef __PARTICLE_INTERFACE_H__
#define __PARTICLE_INTERFACE_H__

#include <SpatialDomains/MeshGraph.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

template <typename T>
inline void expand_bounding_box(T element,
                                std::array<double, 6> &bounding_box) {

  auto element_bounding_box = element->GetBoundingBox();
  for (int dimx = 0; dimx < 3; dimx++) {
    bounding_box[dimx] =
        std::min(bounding_box[dimx], element_bounding_box[dimx]);
    bounding_box[dimx + 3] =
        std::max(bounding_box[dimx + 3], element_bounding_box[dimx + 3]);
  }
}

class ParticleMeshInterface {

private:
public:
  Nektar::SpatialDomains::MeshGraphSharedPtr graph;
  MPI_Comm comm;
  int ndim;
  MeshHierarchy mesh_hierarchy;
  std::array<double, 6> bounding_box;
  std::array<double, 6> global_bounding_box;
  std::array<double, 3> extents;
  std::array<double, 3> global_extents;

  ~ParticleMeshInterface() {}
  ParticleMeshInterface(Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                        const int subdivision_order_offset = 0,
                        MPI_Comm comm = MPI_COMM_WORLD)
      : graph(graph), comm(comm) {

    this->ndim = graph->GetMeshDimension();

    NESOASSERT(graph->GetCurvedEdges().size() == 0,
               "Curved edge found in graph.");
    NESOASSERT(graph->GetCurvedFaces().size() == 0,
               "Curved face found in graph.");
    NESOASSERT(graph->GetAllTetGeoms().size() == 0,
               "Tet element found in graph.");
    NESOASSERT(graph->GetAllPyrGeoms().size() == 0,
               "Pyr element found in graph.");
    NESOASSERT(graph->GetAllPrismGeoms().size() == 0,
               "Prism element found in graph.");
    NESOASSERT(graph->GetAllHexGeoms().size() == 0,
               "Hex element found in graph.");

    auto triangles = graph->GetAllTriGeoms();
    auto quads = graph->GetAllQuadGeoms();

    // Get a local and global bounding box for the mesh
    for (int dimx = 0; dimx < 3; dimx++) {
      this->bounding_box[dimx] = std::numeric_limits<double>::max();
      this->bounding_box[dimx + 3] = std::numeric_limits<double>::min();
    }

    int64_t num_elements = 0;
    for (auto &e : triangles) {
      expand_bounding_box(e.second, this->bounding_box);
      num_elements++;
    }
    for (auto &e : quads) {
      expand_bounding_box(e.second, this->bounding_box);
      num_elements++;
    }

    MPICHK(MPI_Allreduce(this->bounding_box.data(),
                         this->global_bounding_box.data(), 3, MPI_DOUBLE,
                         MPI_MIN, this->comm));
    MPICHK(MPI_Allreduce(this->bounding_box.data() + 3,
                         this->global_bounding_box.data() + 3, 3, MPI_DOUBLE,
                         MPI_MAX, this->comm));

    // Compute a set of coarse mesh sizes and dimensions for the mesh heirarchy
    double min_extent = std::numeric_limits<double>::max();
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const double tmp_global_extent =
          this->global_bounding_box[dimx + 3] - this->global_bounding_box[dimx];
      const double tmp_extent =
          this->bounding_box[dimx + 3] - this->bounding_box[dimx];
      this->extents[dimx] = tmp_extent;
      this->global_extents[dimx] = tmp_global_extent;

      min_extent = std::min(min_extent, tmp_global_extent);
    }
    NESOASSERT(min_extent > 0.0, "Minimum extent is <= 0");

    std::vector<int> dims(this->ndim);
    std::vector<double> origin(this->ndim);

    int64_t cell_count = 1;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      origin[dimx] = this->global_bounding_box[dimx];
      const int tmp_dim = std::ceil(this->global_extents[dimx] / min_extent);
      dims[dimx] = tmp_dim;
      cell_count *= ((int64_t)tmp_dim);
    }

    // compute a subdivision order that would result in the same number of fine
    // cells in the mesh heirarchy as mesh elements in Nektar++
    const double inverse_ndim = 1.0 / ((double)this->ndim);
    const int matching_subdivision_order = std::ceil(
        (((double)std::log(num_elements)) - ((double)std::log(cell_count))) *
        inverse_ndim);

    // apply the offset to this order and compute the used subdivision order
    const int subdivision_order =
        std::max(0, matching_subdivision_order + subdivision_order_offset);

    // create the mesh hierarchy
    this->mesh_hierarchy = MeshHierarchy(this->comm, this->ndim, dims, origin,
                                         min_extent, subdivision_order);
  }
};

#endif
