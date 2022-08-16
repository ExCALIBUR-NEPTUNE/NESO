#ifndef __PARTICLE_INTERFACE_H__
#define __PARTICLE_INTERFACE_H__

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>
#include <array>
#include <limits>
#include <algorithm>
#include <mpi.h>

using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;


template <typename T>
inline void expand_bounding_box(
  T element,
  std::array<double, 6> &bounding_box
){

  auto element_bounding_box = element->GetBoundingBox();
  for(int dimx=0 ; dimx<3 ; dimx++){
    bounding_box[dimx] = std::min(bounding_box[dimx], element_bounding_box[dimx]);
    bounding_box[dimx+3] = std::max(bounding_box[dimx+3], element_bounding_box[dimx+3]);
  }

}



class ParticleMeshInterface {

private:
public:
  Nektar::SpatialDomains::MeshGraphSharedPtr graph;
  MPI_Comm comm;
  int ndim;
  std::array<double, 6> bounding_box;
  std::array<double, 6> global_bounding_box;

  ~ParticleMeshInterface() {}
  ParticleMeshInterface(Nektar::SpatialDomains::MeshGraphSharedPtr graph,
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
    
    for(int dimx=0 ; dimx<3 ; dimx++){
      this->bounding_box[dimx] = std::numeric_limits<double>::max();
      this->bounding_box[dimx+3] = std::numeric_limits<double>::min();
    }

    for(auto &e : triangles){
      expand_bounding_box(e.second, this->bounding_box);
    }
    for(auto &e : quads){
      expand_bounding_box(e.second, this->bounding_box);
    }
    
    MPICHK(MPI_Allreduce(this->bounding_box.data(), this->global_bounding_box.data(), 3,
                  MPI_DOUBLE, MPI_MIN, this->comm));
    MPICHK(MPI_Allreduce(this->bounding_box.data()+3, this->global_bounding_box.data()+3, 3,
                  MPI_DOUBLE, MPI_MAX, this->comm));




  }
};

#endif
