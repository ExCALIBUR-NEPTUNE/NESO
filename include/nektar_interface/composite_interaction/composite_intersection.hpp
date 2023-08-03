#ifndef __COMPOSITE_INTERSECTION_H_
#define __COMPOSITE_INTERSECTION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>

#include <map>
#include <utility>
#include <vector>
#include <set>

namespace NESO::CompositeInteraction {

/**
 *  High-level class to detect and compute the intersection of a particle
 *  trajectory and a Nektar++ composite.
 */
class CompositeIntersection {
protected:
  const int ndim;

public:
  /// Disable (implicit) copies.
  CompositeIntersection(const CompositeIntersection &st) = delete;
  /// Disable (implicit) copies.
  CompositeIntersection &operator=(CompositeIntersection const &a) = delete;

  /// The NESO::Particles Sym<REAL> used to store the previous particle
  /// position.
  const static inline Sym<REAL> previous_position_sym =
      Sym<REAL>("NESO_COMP_INT_PREV_POS");

  /// The composite indices for which the class detects intersections with.
  const std::vector<int> composite_indices;

  /**
   *  TODO
   */
  CompositeIntersection(ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                        std::vector<int> &composite_indices)
      : ndim(particle_mesh_interface->graph->GetMeshDimension()), composite_indices(composite_indices) {
    
    auto graph = particle_mesh_interface->graph;
    // map from composite indices to CompositeSharedPtr
    auto graph_composites = graph->GetComposites();

    // Vector of geometry objects and the composite they correspond to
    std::vector<std::pair<SpatialDomains::GeometrySharedPtr, int>>
        geometry_composites;

    for (auto ix : composite_indices) {
      // check the composite of interest exists in the MeshGraph
      NESOASSERT(graph_composites.count(ix),
                 "Could not find composite index in MeshGraph");
      // each composite has a vector of GeometrySharedPtrs in the m_geomVec
      // attribute
      auto geoms = graph_composites.at(ix)->m_geomVec;
      geometry_composites.reserve(geoms.size() + geometry_composites.size());
      for (auto &geom : geoms) {
        auto shape_type = geom->GetShapeType();
        NESOASSERT(shape_type == LibUtilities::eTriangle ||
                       shape_type == LibUtilities::eQuadrilateral,
                   "unknown composite shape type");
        geometry_composites.push_back({geom, ix});
      }
    }

    // map from mesh hierarchy cells to the packed geoms for that cell
    std::map<INT, std::vector<unsigned char>> packed_geoms;

    // pack each geometry object and reuse the local_id which originally stored
    // local neso cell index to store composite index
    set<int> send_ranks_set;

    std::map<int, std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::Geometry2D>>>> rank_element_map;

    for (auto geom_pair : geometry_composites) {
      // pack the geom and composite into a unsigned char buffer
      SpatialDomains::GeometrySharedPtr geom = geom_pair.first;
      const int composite = geom_pair.second;
      auto geom_2d =
          std::dynamic_pointer_cast<SpatialDomains::Geometry2D>(geom);
      
      //GeometryTransport::PackedGeom2D packed_geom_2d(0, composite, geom_2d);

      // find all mesh hierarchy cells the geom intersects with
      std::deque<std::pair<INT, double>> cells;
      bounding_box_map(geom_2d, particle_mesh_interface->mesh_hierarchy, cells, 0.02);
      auto rgeom_2d = std::make_shared<RemoteGeom2D<SpatialDomains::Geometry2D>>(
        0, composite, geom_2d
      );
      for (auto cell_overlap : cells){
        const INT cell = cell_overlap.first;
        //packed_geoms[cell].insert(std::end(packed_geoms[cell]), 
        //  std::begin(packed_geom_2d.buf), std::end(packed_geom_2d.buf));
        const int owning_rank = particle_mesh_interface->mesh_hierarchy->get_owner(cell);
        send_ranks_set.insert(owning_rank);
        rank_element_map[owning_rank].push_back(rgeom_2d);
      }
    }

    std::vector<int> send_ranks{};
    send_ranks.reserve(send_ranks_set.size());
    for(int rankx : send_ranks_set){
      send_ranks.push_back(rankx);
    }
    



  }

  /**
   *  Method to store the current particle positions before an integration step.
   *
   *  @param particle_group Particles to store current positions of.
   */
  inline void pre_integration(ParticleGroupSharedPtr particle_group) {
    const auto position_dat = particle_group->position_dat;
    const int ndim = position_dat->ncomp;
    NESOASSERT(ndim == this->ndim,
               "missmatch between particle ndim and class ndim");
    const auto sycl_target = particle_group->sycl_target;
    // If the previous position dat does not already exist create it here
    if (!particle_group->contains_dat(previous_position_sym)) {
      particle_group->add_particle_dat(
          ParticleDat(sycl_target, ParticleProp(previous_position_sym, ndim),
                      particle_group->domain->mesh->get_cell_count()));
    }

    // copy the current position onto the previous position
    auto pl_iter_range = position_dat->get_particle_loop_iter_range();
    auto pl_stride = position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = position_dat->get_particle_loop_npart_cell();
    const auto k_P = position_dat->cell_dat.device_ptr();
    auto k_PP =
        particle_group->get_dat(previous_position_sym)->cell_dat.device_ptr();
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                for (int dimx = 0; dimx < ndim; dimx++) {
                  k_PP[cellx][dimx][layerx] = k_P[cellx][dimx][layerx];
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
  }
};

} // namespace NESO::CompositeInteraction

#endif
