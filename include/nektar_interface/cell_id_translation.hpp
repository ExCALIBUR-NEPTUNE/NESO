#ifndef __CELL_ID_TRANSLATION_H__
#define __CELL_ID_TRANSLATION_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <vector>

#include <mpi.h>

#include "particle_mesh_interface.hpp"
#include <SpatialDomains/MeshGraph.h>

#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

namespace NESO {

/**
 *  Class to convert Nektar++ global ids of geometry objects to ids that can be
 *  used by NESO-Particles.
 */
class CellIDTranslation {
private:
  ParticleDatSharedPtr<INT> cell_id_dat;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  BufferDeviceHost<int> id_map;
  int shift;

  template <typename T>
  inline void construct_maps(std::map<int, std::shared_ptr<T>> &geoms) {
    const int nelements = geoms.size();
    int id_min = std::numeric_limits<int>::max();
    int id_max = std::numeric_limits<int>::min();

    this->map_to_nektar.resize(nelements);
    this->dh_map_to_geom_type.realloc_no_copy(nelements);
    for (int cellx = 0; cellx < nelements; cellx++) {
      this->dh_map_to_geom_type.h_buffer.ptr[cellx] = -1;
    }

    int index = 0;
    for (auto &geom : geoms) {
      const int id = geom.second->GetGlobalID();
      NESOASSERT(geom.first == id, "Expected these ids to match");
      id_min = std::min(id_min, id);
      id_max = std::max(id_max, id);
      this->map_to_nektar[index] = id;
      // record the type of this cell.
      this->dh_map_to_geom_type.h_buffer.ptr[index] =
          shape_type_to_int(geom.second->GetShapeType());
      index++;
    }

    NESOASSERT(index == nelements, "element count missmatch");
    this->shift = id_min;
    const int shifted_max = id_max - id_min;
    id_map.realloc_no_copy(shifted_max + 1);

    for (int ex = 0; ex < nelements; ex++) {
      const int lookup_index = this->map_to_nektar[ex] - this->shift;
      this->id_map.h_buffer.ptr[lookup_index] = ex;
    }
    this->id_map.host_to_device();
    this->dh_map_to_geom_type.host_to_device();
  }

public:
  ~CellIDTranslation(){};

  /// The sycl target this map exists on.
  SYCLTargetSharedPtr sycl_target;

  /// Map from NESO-Particles ids to nektar++ global ids.
  std::vector<int> map_to_nektar;

  /**
   * Map from NESO-Particles ids to nektar++ Geom type where:
   * 0 -> TriGeom
   * 1 -> QuadGeom
   */
  BufferDeviceHost<int> dh_map_to_geom_type;

  /**
   * Create a new geometry id mapper.
   *
   * @param sycl_target Compute device to use.
   * @param cell_id_dat ParticleDat of cell ids.
   * @param particle_mesh_interface Interface object between Nektar++ graph and
   * NESO-Particles.
   */
  CellIDTranslation(SYCLTargetSharedPtr sycl_target,
                    ParticleDatSharedPtr<INT> cell_id_dat,
                    ParticleMeshInterfaceSharedPtr particle_mesh_interface)
      : sycl_target(sycl_target), cell_id_dat(cell_id_dat),
        particle_mesh_interface(particle_mesh_interface),
        id_map(sycl_target, 1), dh_map_to_geom_type(sycl_target, 1) {

    auto graph = this->particle_mesh_interface->graph;
    const int ndim = particle_mesh_interface->ndim;
    if (ndim == 2) {
      std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
          geoms_2d;
      get_all_elements_2d(graph, geoms_2d);
      this->construct_maps(geoms_2d);
    } else if (ndim == 3) {
      std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
          geoms_3d;
      get_all_elements_3d(graph, geoms_3d);
      this->construct_maps(geoms_3d);
    } else {
      NESOASSERT(false, "Unsupported spatial dimension.");
    }
  };

  /**
   *  Loop over all particles and map cell ids from Nektar++ cell ids to
   *  NESO-Particle cells ids.
   */
  inline void execute() {
    auto t0 = profile_timestamp();

    auto pl_iter_range = this->cell_id_dat->get_particle_loop_iter_range();
    auto pl_stride = this->cell_id_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = this->cell_id_dat->get_particle_loop_npart_cell();

    auto k_cell_id_dat = this->cell_id_dat->cell_dat.device_ptr();
    const auto k_lookup_map = this->id_map.d_buffer.ptr;
    const INT k_shift = this->shift;

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const INT nektar_cell = k_cell_id_dat[cellx][0][layerx];
                const INT shifted_nektar_cell = nektar_cell - k_shift;
                const INT neso_cell = k_lookup_map[shifted_nektar_cell];
                k_cell_id_dat[cellx][0][layerx] = neso_cell;

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("CellIDTranslation", "execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };
};

typedef std::shared_ptr<CellIDTranslation> CellIDTranslationSharedPtr;

} // namespace NESO

#endif
