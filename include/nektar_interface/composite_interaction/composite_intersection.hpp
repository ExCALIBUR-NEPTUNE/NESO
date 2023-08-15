#ifndef __COMPOSITE_INTERSECTION_H_
#define __COMPOSITE_INTERSECTION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>

#include "composite_transport.hpp"

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace NESO::CompositeInteraction {

namespace {

inline int indexing_cell_min(const int cell, const int num_cells, const int dim,
                             const int ndim) {
  return cell + dim * num_cells;
}

inline int indexing_cell_max(const int cell, const int num_cells, const int dim,
                             const int ndim) {
  return cell + dim * num_cells + ndim * num_cells;
}

} // namespace

/**
 *  High-level class to detect and compute the intersection of a particle
 *  trajectory and a Nektar++ composite.
 */
class CompositeIntersection {
protected:
  const int ndim;
  const int num_cells;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  std::unique_ptr<CompositeTransport> composite_transport;
  std::unique_ptr<BufferDevice<INT>> d_cell_min_maxes;
  std::unique_ptr<MeshHierarchyMapper> mesh_hierarchy_mapper;
  std::unique_ptr<BufferDeviceHost<INT>> dh_max_bounding_box_size;
  std::unique_ptr<BufferDeviceHost<INT>> dh_mh_cells;
  std::unique_ptr<BufferDeviceHost<int>> dh_mh_cells_index;

  inline void find_cells(ParticleGroupSharedPtr particle_group,
                         std::set<INT> &cells) {
    NESOASSERT(this->ndim < 4,
               "Method assumes no more than 3 spatial dimensions.");
    const auto position_dat = particle_group->position_dat;
    const int k_ndim = this->ndim;

    // copy the current position onto the previous position
    auto pl_iter_range = position_dat->get_particle_loop_iter_range();
    auto pl_stride = position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = position_dat->get_particle_loop_npart_cell();

    const auto k_P = position_dat->cell_dat.device_ptr();
    const auto k_PP =
        particle_group->get_dat(previous_position_sym)->cell_dat.device_ptr();

    const auto mesh_hierarchy_device_mapper =
        mesh_hierarchy_mapper->get_device_mapper();

    const int k_num_cells = this->num_cells;
    const INT k_INT_MAX = std::numeric_limits<INT>::max();
    const INT k_INT_MIN = std::numeric_limits<INT>::min();
    auto k_cell_min_maxes = this->d_cell_min_maxes->ptr;

    const auto d_npart_cell = position_dat->d_npart_cell;

    // reset the bounding mesh hierarchy boxes for each element
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(k_num_cells), [=](sycl::id<1> idx) {
            for (int dimx = 0; dimx < k_ndim; dimx++) {
              k_cell_min_maxes[indexing_cell_min(idx, k_num_cells, dimx,
                                                 k_ndim)] = k_INT_MAX;

              k_cell_min_maxes[indexing_cell_max(idx, k_num_cells, dimx,
                                                 k_ndim)] = k_INT_MIN;
            }
          });
        })
        .wait_and_throw();

    // compute the new bounding boxes of mesh hierarchies for each element
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                REAL position[3];
                INT cell_cart[3];

                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  position[dimx] = k_P[cellx][dimx][layerx];
                }
                mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(
                    position, cell_cart);

                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  {
                    sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        ar(k_cell_min_maxes[indexing_cell_min(
                            cellx, k_num_cells, dimx, k_ndim)]);
                    ar.fetch_min(cell_cart[dimx]);
                  }
                  {
                    sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        ar(k_cell_min_maxes[indexing_cell_max(
                            cellx, k_num_cells, dimx, k_ndim)]);
                    ar.fetch_max(cell_cart[dimx]);
                  }
                }

                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  position[dimx] = k_PP[cellx][dimx][layerx];
                }
                mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(
                    position, cell_cart);

                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  {
                    sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        ar(k_cell_min_maxes[indexing_cell_min(
                            cellx, k_num_cells, dimx, k_ndim)]);
                    ar.fetch_min(cell_cart[dimx]);
                  }
                  {
                    sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        ar(k_cell_min_maxes[indexing_cell_max(
                            cellx, k_num_cells, dimx, k_ndim)]);
                    ar.fetch_max(cell_cart[dimx]);
                  }
                }

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    this->dh_max_bounding_box_size->h_buffer.ptr[0] = 0;
    this->dh_max_bounding_box_size->host_to_device();
    auto k_max_ptr = this->dh_max_bounding_box_size->d_buffer.ptr;

    // determine the maximum bounding box size
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(k_num_cells), [=](sycl::id<1> idx) {
            if (d_npart_cell[idx] > 0) {
              INT volume = 1;
              for (int dimx = 0; dimx < k_ndim; dimx++) {
                const INT bound_min = k_cell_min_maxes[indexing_cell_min(
                    idx, k_num_cells, dimx, k_ndim)];

                const INT bound_max = k_cell_min_maxes[indexing_cell_max(
                    idx, k_num_cells, dimx, k_ndim)];

                const int width = bound_max - bound_min + 1;
                volume *= width;
              }

              sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                               sycl::memory_scope::device>
                  ar(k_max_ptr[0]);
              ar.fetch_max(volume);
            }
          });
        })
        .wait_and_throw();

    // realloc the array storing the cells covered if needed
    this->dh_max_bounding_box_size->device_to_host();
    const INT max_bounding_box_size =
        this->dh_max_bounding_box_size->h_buffer.ptr[0];
    const INT required_cells_array_size = max_bounding_box_size * num_cells;
    if (this->dh_mh_cells->size < required_cells_array_size) {
      this->dh_mh_cells->realloc_no_copy(required_cells_array_size);
    }

    // get the cells covered as linear mesh hierarchy indices
    this->dh_mh_cells_index->h_buffer.ptr[0] = 0;
    this->dh_mh_cells_index->host_to_device();
    auto k_mh_cells_index = this->dh_mh_cells_index->d_buffer.ptr;
    auto k_mh_cells = this->dh_mh_cells->d_buffer.ptr;
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(k_num_cells), [=](sycl::id<1> idx) {
            if (d_npart_cell[idx] > 0) {
              INT cell_starts[3] = {0, 0, 0};
              INT cell_ends[3] = {1, 1, 1};

              // sanitise the bounds to actually be in the domain
              for (int dimx = 0; dimx < k_ndim; dimx++) {
                const INT bound_min = k_cell_min_maxes[indexing_cell_min(
                    idx, k_num_cells, dimx, k_ndim)];

                const INT bound_max = k_cell_min_maxes[indexing_cell_max(
                    idx, k_num_cells, dimx, k_ndim)];

                const INT max_possible_cell =
                    mesh_hierarchy_device_mapper.dims[dimx] *
                    mesh_hierarchy_device_mapper.ncells_dim_fine;

                if ((bound_min >= 0) && (bound_min < max_possible_cell)) {
                  cell_starts[dimx] = bound_min;
                }

                if ((bound_max >= 0) && (bound_max < max_possible_cell)) {
                  cell_ends[dimx] = bound_max + 1;
                }
              }

              // loop over the cells in the bounding box
              INT cell_index[3];
              for (cell_index[2] = cell_starts[2]; cell_index[2] < cell_ends[2];
                   cell_index[2]++) {
                for (cell_index[1] = cell_starts[1];
                     cell_index[1] < cell_ends[1]; cell_index[1]++) {
                  for (cell_index[0] = cell_starts[0];
                       cell_index[0] < cell_ends[0]; cell_index[0]++) {

                    // convert the cartesian cell index into a mesh heirarchy
                    // index
                    INT mh_tuple[6];
                    mesh_hierarchy_device_mapper.cart_tuple_to_tuple(cell_index,
                                                                     mh_tuple);
                    // convert the mesh hierarchy tuple to linear index
                    const INT linear_index =
                        mesh_hierarchy_device_mapper.tuple_to_linear_global(
                            mh_tuple);

                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        ar(k_mh_cells_index[0]);
                    const int index = ar.fetch_add(1);
                    k_mh_cells[index] = linear_index;
                  }
                }
              }
            }
          });
        })
        .wait_and_throw();

    // collect the mesh hierarchy cells on the host and remove duplicates
    this->dh_mh_cells->device_to_host();
    this->dh_mh_cells_index->device_to_host();
    const int num_collect_mh_cells = this->dh_mh_cells_index->h_buffer.ptr[0];
    cells.clear();
    for (int cx = 0; cx < num_collect_mh_cells; cx++) {
      const INT cell = this->dh_mh_cells->h_buffer.ptr[cx];
      cells.insert(cell);
    }
  }

public:
  /// Disable (implicit) copies.
  CompositeIntersection(const CompositeIntersection &st) = delete;
  /// Disable (implicit) copies.
  CompositeIntersection &operator=(CompositeIntersection const &a) = delete;

  /// SYCLTarget to use for computation.
  SYCLTargetSharedPtr sycl_target;

  /// The NESO::Particles Sym<REAL> used to store the previous particle
  /// position.
  const static inline Sym<REAL> previous_position_sym =
      Sym<REAL>("NESO_COMP_INT_PREV_POS");

  /// The composite indices for which the class detects intersections with.
  const std::vector<int> composite_indices;

  /**
   * TODO
   */
  inline void free() { this->composite_transport->free(); }

  /**
   *  TODO
   */
  CompositeIntersection(SYCLTargetSharedPtr sycl_target,
                        ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                        std::vector<int> &composite_indices)
      : sycl_target(sycl_target),
        particle_mesh_interface(particle_mesh_interface),
        ndim(particle_mesh_interface->graph->GetMeshDimension()),
        composite_indices(composite_indices),
        num_cells(particle_mesh_interface->get_cell_count()) {

    this->composite_transport = std::make_unique<CompositeTransport>(
        particle_mesh_interface, composite_indices);
    this->mesh_hierarchy_mapper = std::make_unique<MeshHierarchyMapper>(
        sycl_target, this->particle_mesh_interface->get_mesh_hierarchy());

    this->d_cell_min_maxes = std::make_unique<BufferDevice<INT>>(
        this->sycl_target, 2 * this->ndim * this->num_cells);

    this->dh_max_bounding_box_size =
        std::make_unique<BufferDeviceHost<INT>>(this->sycl_target, 1);
    this->dh_mh_cells =
        std::make_unique<BufferDeviceHost<INT>>(this->sycl_target, 128);
    this->dh_mh_cells_index =
        std::make_unique<BufferDeviceHost<int>>(this->sycl_target, 1);
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
    NESOASSERT(this->sycl_target == particle_group->sycl_target,
               "missmatch of sycl target");

    // If the previous position dat does not already exist create it here
    if (!particle_group->contains_dat(previous_position_sym)) {
      particle_group->add_particle_dat(ParticleDat(
          this->sycl_target, ParticleProp(previous_position_sym, ndim),
          particle_group->domain->mesh->get_cell_count()));
    }

    // copy the current position onto the previous position
    auto pl_iter_range = position_dat->get_particle_loop_iter_range();
    auto pl_stride = position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = position_dat->get_particle_loop_npart_cell();
    const auto k_P = position_dat->cell_dat.device_ptr();
    auto k_PP =
        particle_group->get_dat(previous_position_sym)->cell_dat.device_ptr();
    this->sycl_target->queue
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

  inline void execute(ParticleGroupSharedPtr particle_group) {
    NESOASSERT(
        particle_group->contains_dat(previous_position_sym),
        "Previous position ParticleDat not found. Was pre_integration called?");

    // find the MeshHierarchy cells that the particles potentially pass though
    std::set<INT> mh_cells;
    this->find_cells(particle_group, mh_cells);
    // Collect the geometry objects for the composites of interest for these
    // cells. On exit from this function mh_cells contains only the new mesh
    // hierarchy cells which were collected.
    this->composite_transport->collect_geometry(mh_cells);
  }
};

} // namespace NESO::CompositeInteraction

#endif
