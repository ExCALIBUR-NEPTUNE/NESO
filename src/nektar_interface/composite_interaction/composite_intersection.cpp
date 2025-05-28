#include <nektar_interface/composite_interaction/composite_intersection.hpp>

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

inline REAL radius_squared(const int ndim, const REAL *r0, const REAL *r1) {
  REAL dd = 0.0;
  for (int dimx = 0; dimx < ndim; dimx++) {
    const REAL d = (r0 - r1);
    const REAL d2 = d * d;
    dd += d2;
  }
  return dd;
}

} // namespace

template <typename T>
void CompositeIntersection::find_cells(std::shared_ptr<T> iteration_set,
                                       std::set<INT> &cells) {
  this->check_iteration_set(iteration_set);

  NESOASSERT(this->ndim < 4,
             "Method assumes no more than 3 spatial dimensions.");

  auto particle_group = this->get_particle_group(iteration_set);
  const auto position_dat = particle_group->position_dat;
  const int k_ndim = this->ndim;

  const auto mesh_hierarchy_device_mapper =
      this->mesh_hierarchy_mapper->get_device_mapper();

  const int k_num_cells = this->num_cells;
  const int k_INT_MAX = std::numeric_limits<int>::max();
  const int k_INT_MIN = std::numeric_limits<int>::min();
  auto k_cell_min_maxes = this->d_cell_min_maxes->ptr;

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
  particle_loop(
      "CompositeIntersection::find_cells", iteration_set,
      [=](auto index, auto k_P, auto k_PP) {
        const INT cellx = index.cell;

        auto lambda_set_min_max =
            [](const auto &cellx, const auto &k_num_cells, const auto &k_ndim,
               const auto &position, const auto &cell_cart,
               auto &k_cell_min_maxes,
               const auto &mesh_hierarchy_device_mapper) -> void {
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            {
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device>
                  ar(k_cell_min_maxes[indexing_cell_min(cellx, k_num_cells,
                                                        dimx, k_ndim)]);

              const int trunc =
                  KERNEL_MAX(0, static_cast<int>(cell_cart[dimx]));
              ar.fetch_min(trunc);
            }
            {
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device>
                  ar(k_cell_min_maxes[indexing_cell_max(cellx, k_num_cells,
                                                        dimx, k_ndim)]);

              const INT max_possible_cell =
                  mesh_hierarchy_device_mapper.dims[dimx] *
                  mesh_hierarchy_device_mapper.ncells_dim_fine;
              const int trunc = KERNEL_MIN(max_possible_cell - 1,
                                           static_cast<int>(cell_cart[dimx]));
              ar.fetch_max(trunc);
            }
          }
        };

        REAL position[3];
        INT cell_cart[3];

        for (int dimx = 0; dimx < k_ndim; dimx++) {
          position[dimx] = k_P.at(dimx);
        }
        mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(position,
                                                                cell_cart);

        lambda_set_min_max(cellx, k_num_cells, k_ndim, position, cell_cart,
                           k_cell_min_maxes, mesh_hierarchy_device_mapper);

        for (int dimx = 0; dimx < k_ndim; dimx++) {
          position[dimx] = k_PP.at(dimx);
        }
        mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(position,
                                                                cell_cart);

        lambda_set_min_max(cellx, k_num_cells, k_ndim, position, cell_cart,
                           k_cell_min_maxes, mesh_hierarchy_device_mapper);
      },
      Access::read(ParticleLoopIndex{}), Access::read(position_dat->sym),
      Access::read(previous_position_sym))
      ->execute();

  this->dh_max_bounding_box_size->h_buffer.ptr[0] = 0;
  this->dh_max_bounding_box_size->host_to_device();
  auto k_max_ptr = this->dh_max_bounding_box_size->d_buffer.ptr;

  // determine the maximum bounding box size
  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(k_num_cells), [=](sycl::id<1> idx) {
          bool valid = true;
          int volume = 1;
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            const int bound_min = k_cell_min_maxes[indexing_cell_min(
                idx, k_num_cells, dimx, k_ndim)];

            const int bound_max = k_cell_min_maxes[indexing_cell_max(
                idx, k_num_cells, dimx, k_ndim)];

            valid = (bound_max < bound_min) ? false : valid;

            const int width = bound_max - bound_min + 1;
            volume *= width;
          }
          if (valid) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
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
          bool valid = true;
          INT cell_starts[3] = {0, 0, 0};
          INT cell_ends[3] = {1, 1, 1};

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            const INT bound_min = k_cell_min_maxes[indexing_cell_min(
                idx, k_num_cells, dimx, k_ndim)];

            const INT bound_max = k_cell_min_maxes[indexing_cell_max(
                idx, k_num_cells, dimx, k_ndim)];

            valid = (bound_max < bound_min) ? false : valid;
            cell_starts[dimx] = bound_min;
            cell_ends[dimx] = bound_max + 1;
          }

          if (valid) {
            // loop over the cells in the bounding box
            INT cell_index[3];
            for (cell_index[2] = cell_starts[2]; cell_index[2] < cell_ends[2];
                 cell_index[2]++) {
              for (cell_index[1] = cell_starts[1]; cell_index[1] < cell_ends[1];
                   cell_index[1]++) {
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

template <typename T>
void CompositeIntersection::find_intersections_2d(
    std::shared_ptr<T> iteration_set, ParticleDatSharedPtr<INT> dat_composite,
    ParticleDatSharedPtr<REAL> dat_positions) {
  this->check_iteration_set(iteration_set);
  auto particle_group = this->get_particle_group(iteration_set);
  NESOASSERT(this->ndim == 2, "Method assumes 2 spatial dimensions.");
  NESOASSERT(dat_positions->ncomp == this->ndim,
             "Missmatch in number of spatial dimensions.");
  NESOASSERT(
      dat_composite->ncomp > 2,
      "Require at least three components for the dat_composite argument.");

  const auto position_dat = particle_group->position_dat;
  const int k_ndim = this->ndim;
  const auto mesh_hierarchy_device_mapper =
      this->mesh_hierarchy_mapper->get_device_mapper();

  const REAL k_REAL_MAX = std::numeric_limits<REAL>::max();

  // the binary map containing the geometry information
  auto k_MAP_ROOT = this->composite_collections->map_cells_collections->root;
  if (k_MAP_ROOT != nullptr) {

    const double k_tol = this->line_intersection_tol;
    const int k_max_iterations = this->newton_max_iteration;

    particle_loop(
        "CompositeIntersection::find_intersections_2d", iteration_set,
        [=](auto k_P, auto k_PP, auto k_OUT_P, auto k_OUT_C) {
          REAL prev_position[2] = {0};
          REAL position[2] = {0};
          INT prev_cell_cart[2] = {0};
          INT cell_cart[2] = {0};

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            position[dimx] = k_P.at(dimx);
          }
          mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(position,
                                                                  cell_cart);

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            prev_position[dimx] = k_PP.at(dimx);
          }
          mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(
              prev_position, prev_cell_cart);

          REAL intersection_distance = k_REAL_MAX;

          INT cell_starts[2] = {0, 0};
          INT cell_ends[2] = {1, 1};

          // sanitise the bounds to actually be in the domain
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            const INT max_possible_cell =
                mesh_hierarchy_device_mapper.dims[dimx] *
                mesh_hierarchy_device_mapper.ncells_dim_fine;

            cell_ends[dimx] = max_possible_cell;

            const INT bound_min =
                KERNEL_MIN(prev_cell_cart[dimx], cell_cart[dimx]);
            const INT bound_max =
                KERNEL_MAX(prev_cell_cart[dimx], cell_cart[dimx]);

            if ((bound_min >= 0) && (bound_min < max_possible_cell)) {
              cell_starts[dimx] = bound_min;
            }

            if ((bound_max >= 0) && (bound_max < max_possible_cell)) {
              cell_ends[dimx] = bound_max + 1;
            }
          }

          REAL i0, i1;
          const REAL p00 = prev_position[0];
          const REAL p01 = prev_position[1];
          const REAL p10 = position[0];
          const REAL p11 = position[1];

          // loop over the cells in the bounding box
          INT cell_index[2];
          for (cell_index[1] = cell_starts[1]; cell_index[1] < cell_ends[1];
               cell_index[1]++) {
            for (cell_index[0] = cell_starts[0]; cell_index[0] < cell_ends[0];
                 cell_index[0]++) {

              // convert the cartesian cell index into a mesh heirarchy
              // index
              INT mh_tuple[4];
              mesh_hierarchy_device_mapper.cart_tuple_to_tuple(cell_index,
                                                               mh_tuple);
              // convert the mesh hierarchy tuple to linear index
              const INT linear_index =
                  mesh_hierarchy_device_mapper.tuple_to_linear_global(mh_tuple);

              // now we actually have a MeshHierarchy linear index to
              // test for composite geoms
              CompositeCollection *cc;
              const bool cell_exists = k_MAP_ROOT->get(linear_index, &cc);

              if (cell_exists) {
                const int num_segments = cc->num_segments;
                for (int sx = 0; sx < num_segments; sx++) {
                  REAL i0, i1;
                  const bool contained =
                      cc->lli_segments[sx].line_line_intersection(
                          p00, p01, p10, p11, &i0, &i1, k_tol);

                  if (contained) {
                    const REAL r0 = p00 - i0;
                    const REAL r1 = p01 - i1;
                    const REAL d2 = r0 * r0 + r1 * r1;
                    if (d2 < intersection_distance) {
                      k_OUT_P.at(0) = i0;
                      k_OUT_P.at(1) = i1;
                      k_OUT_C.at(0) = cc->group_ids_segments[sx];
                      k_OUT_C.at(1) = cc->composite_ids_segments[sx];
                      k_OUT_C.at(2) = cc->geom_ids_segments[sx];
                      intersection_distance = d2;
                    }
                  }
                }
              }
            }
          }
        },
        Access::read(position_dat->sym), Access::read(previous_position_sym),
        Access::write(dat_positions->sym), Access::write(dat_composite->sym))
        ->execute();
  }
}

template <typename T>
void CompositeIntersection::find_intersections_3d(
    std::shared_ptr<T> iteration_set, ParticleDatSharedPtr<INT> dat_composite,
    ParticleDatSharedPtr<REAL> dat_positions) {
  this->check_iteration_set(iteration_set);
  auto particle_group = this->get_particle_group(iteration_set);
  NESOASSERT(this->ndim == 3, "Method assumes 3 spatial dimensions.");
  NESOASSERT(dat_positions->ncomp == this->ndim,
             "Missmatch in number of spatial dimensions.");
  NESOASSERT(
      dat_composite->ncomp > 2,
      "Require at least three components for the dat_composite argument.");

  const auto position_dat = particle_group->position_dat;
  const int k_ndim = this->ndim;
  const auto mesh_hierarchy_device_mapper =
      this->mesh_hierarchy_mapper->get_device_mapper();

  const REAL k_REAL_MAX = std::numeric_limits<REAL>::max();

  // the binary map containing the geometry information
  auto k_MAP_ROOT = this->composite_collections->map_cells_collections->root;
  if (k_MAP_ROOT != nullptr) {

    const double k_newton_tol = this->newton_tol;
    const double k_contained_tol = this->contained_tol;
    const int k_max_iterations = this->newton_max_iteration;
    const auto k_MASK = this->mask;
    const int grid_size = std::max(
        this->num_modes_factor * this->composite_collections->max_num_modes - 1,
        1);
    const int k_grid_size_x = grid_size;
    const int k_grid_size_y = grid_size;
    const REAL k_grid_width = 2.0 / grid_size;

    static_assert(!Newton::local_memory_required<
                      Newton::MappingQuadLinear2DEmbed3D>::required,
                  "Did not expect local memory to be required for this Newton "
                  "implementation");

    particle_loop(
        "CompositeIntersection::find_intersections_3d_quads", iteration_set,
        [=](auto k_P, auto k_PP, auto k_OUT_P, auto k_OUT_C) {
          REAL prev_position[3] = {0};
          REAL position[3] = {0};
          INT prev_cell_cart[3] = {0};
          INT cell_cart[3] = {0};

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            position[dimx] = k_P.at(dimx);
          }
          mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(position,
                                                                  cell_cart);

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            prev_position[dimx] = k_PP.at(dimx);
          }
          mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(
              prev_position, prev_cell_cart);

          REAL intersection_distance = k_REAL_MAX;

          INT cell_starts[3] = {0, 0, 0};
          INT cell_ends[3] = {1, 1, 1};

          // sanitise the bounds to actually be in the domain
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            const INT max_possible_cell =
                mesh_hierarchy_device_mapper.dims[dimx] *
                mesh_hierarchy_device_mapper.ncells_dim_fine;

            cell_ends[dimx] = max_possible_cell;

            const INT bound_min =
                KERNEL_MIN(prev_cell_cart[dimx], cell_cart[dimx]);
            const INT bound_max =
                KERNEL_MAX(prev_cell_cart[dimx], cell_cart[dimx]);

            if ((bound_min >= 0) && (bound_min < max_possible_cell)) {
              cell_starts[dimx] = bound_min;
            }

            if ((bound_max >= 0) && (bound_max < max_possible_cell)) {
              cell_ends[dimx] = bound_max + 1;
            }
          }

          REAL i0, i1, i2;
          const REAL p00 = prev_position[0];
          const REAL p01 = prev_position[1];
          const REAL p02 = prev_position[2];
          const REAL p10 = position[0];
          const REAL p11 = position[1];
          const REAL p12 = position[2];

          // loop over the cells in the bounding box
          INT cell_index[3];
          for (cell_index[2] = cell_starts[2]; cell_index[2] < cell_ends[2];
               cell_index[2]++) {
            for (cell_index[1] = cell_starts[1]; cell_index[1] < cell_ends[1];
                 cell_index[1]++) {
              for (cell_index[0] = cell_starts[0]; cell_index[0] < cell_ends[0];
                   cell_index[0]++) {

                // convert the cartesian cell index into a mesh heirarchy
                // index
                INT mh_tuple[6];
                mesh_hierarchy_device_mapper.cart_tuple_to_tuple(cell_index,
                                                                 mh_tuple);
                // convert the mesh hierarchy tuple to linear index
                const INT linear_index =
                    mesh_hierarchy_device_mapper.tuple_to_linear_global(
                        mh_tuple);

                // now we actually have a MeshHierarchy linear index to
                // test for composite geoms
                CompositeCollection *cc;
                const bool cell_exists = k_MAP_ROOT->get(linear_index, &cc);

                if (cell_exists) {
                  const int num_quads = cc->num_quads;
                  REAL eta0, eta1, eta2;
                  for (int gx = 0; gx < num_quads; gx++) {
                    // get the plane of the geom
                    const LinePlaneIntersection *lpi = &cc->lpi_quads[gx];
                    // does the trajectory intersect the plane
                    if (lpi->line_segment_intersection(p00, p01, p02, p10, p11,
                                                       p12, &i0, &i1, &i2)) {
                      // is the intersection point near to the geom
                      if (lpi->point_near_to_geom(i0, i1, i2)) {

                        const Newton::MappingQuadLinear2DEmbed3D::DataDevice
                            *map_data = cc->buf_quads + gx;
                        Newton::MappingNewtonIterationBase<
                            Newton::MappingQuadLinear2DEmbed3D>
                            k_newton_type{};
                        Newton::XMapNewtonKernel<
                            Newton::MappingQuadLinear2DEmbed3D>
                            k_newton_kernel;

                        bool cell_found = false;

                        // Quads don't have a singularity we need to consider
                        for (int g1 = 0; (g1 <= k_grid_size_y) && (!cell_found);
                             g1++) {
                          for (int g0 = 0;
                               (g0 <= k_grid_size_x) && (!cell_found); g0++) {

                            REAL xi[3] = {-1.0 + g0 * k_grid_width,
                                          -1.0 + g1 * k_grid_width, 0.0};

                            const bool converged = k_newton_kernel.x_inverse(
                                map_data, i0, i1, i2, &xi[0], &xi[1], &xi[2],
                                nullptr, k_max_iterations, k_newton_tol, true);
                            k_newton_type.loc_coord_to_loc_collapsed(
                                map_data, xi[0], xi[1], xi[2], &eta0, &eta1,
                                &eta2);

                            eta0 = Kernel::min(eta0, 1.0 + k_contained_tol);
                            eta1 = Kernel::min(eta1, 1.0 + k_contained_tol);
                            eta2 = Kernel::min(eta2, 1.0 + k_contained_tol);
                            eta0 = Kernel::max(eta0, -1.0 - k_contained_tol);
                            eta1 = Kernel::max(eta1, -1.0 - k_contained_tol);
                            eta2 = Kernel::max(eta2, -1.0 - k_contained_tol);

                            k_newton_type.loc_collapsed_to_loc_coord(
                                map_data, eta0, eta1, eta2, &xi[0], &xi[1],
                                &xi[2]);

                            const REAL clamped_residual =
                                k_newton_type.newton_residual(
                                    map_data, xi[0], xi[1], xi[2], i0, i1, i2,
                                    &eta0, &eta1, &eta2, nullptr);

                            const bool contained =
                                clamped_residual <= k_newton_tol;

                            cell_found = contained && converged;
                            if (cell_found) {
                              const REAL r0 = p00 - i0;
                              const REAL r1 = p01 - i1;
                              const REAL r2 = p02 - i2;
                              const REAL d2 = r0 * r0 + r1 * r1 + r2 * r2;
                              if (d2 < intersection_distance) {
                                k_OUT_P.at(0) = i0;
                                k_OUT_P.at(1) = i1;
                                k_OUT_P.at(2) = i2;
                                k_OUT_C.at(0) = cc->group_ids_quads[gx];
                                k_OUT_C.at(1) = cc->composite_ids_quads[gx];
                                k_OUT_C.at(2) = cc->geom_ids_quads[gx];
                                intersection_distance = d2;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        },
        Access::read(position_dat->sym), Access::read(previous_position_sym),
        Access::write(dat_positions->sym), Access::write(dat_composite->sym))
        ->execute();

    static_assert(!Newton::local_memory_required<
                      Newton::MappingTriangleLinear2DEmbed3D>::required,
                  "Did not expect local memory to be required for this Newton "
                  "implemenation");

    particle_loop(
        "CompositeIntersection::find_intersections_3d_triangles", iteration_set,
        [=](auto k_P, auto k_PP, auto k_OUT_P, auto k_OUT_C) {
          REAL prev_position[3] = {0};
          REAL position[3] = {0};
          INT prev_cell_cart[3] = {0};
          INT cell_cart[3] = {0};

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            position[dimx] = k_P.at(dimx);
          }
          mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(position,
                                                                  cell_cart);

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            prev_position[dimx] = k_PP.at(dimx);
          }
          mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(
              prev_position, prev_cell_cart);

          const REAL d0 = k_OUT_P.at(0) - k_PP.at(0);
          const REAL d1 = k_OUT_P.at(1) - k_PP.at(1);
          const REAL d2 = k_OUT_P.at(2) - k_PP.at(2);
          const REAL intersection_distance_p = d0 * d0 + d1 * d1 + d2 * d2;

          REAL intersection_distance =
              (k_OUT_C.at(0) != k_MASK) ? intersection_distance_p : k_REAL_MAX;

          INT cell_starts[3] = {0, 0, 0};
          INT cell_ends[3] = {1, 1, 1};

          // sanitise the bounds to actually be in the domain
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            const INT max_possible_cell =
                mesh_hierarchy_device_mapper.dims[dimx] *
                mesh_hierarchy_device_mapper.ncells_dim_fine;

            cell_ends[dimx] = max_possible_cell;

            const INT bound_min =
                KERNEL_MIN(prev_cell_cart[dimx], cell_cart[dimx]);
            const INT bound_max =
                KERNEL_MAX(prev_cell_cart[dimx], cell_cart[dimx]);

            if ((bound_min >= 0) && (bound_min < max_possible_cell)) {
              cell_starts[dimx] = bound_min;
            }

            if ((bound_max >= 0) && (bound_max < max_possible_cell)) {
              cell_ends[dimx] = bound_max + 1;
            }
          }

          REAL i0, i1, i2;
          const REAL p00 = prev_position[0];
          const REAL p01 = prev_position[1];
          const REAL p02 = prev_position[2];
          const REAL p10 = position[0];
          const REAL p11 = position[1];
          const REAL p12 = position[2];

          // loop over the cells in the bounding box
          INT cell_index[3];
          for (cell_index[2] = cell_starts[2]; cell_index[2] < cell_ends[2];
               cell_index[2]++) {
            for (cell_index[1] = cell_starts[1]; cell_index[1] < cell_ends[1];
                 cell_index[1]++) {
              for (cell_index[0] = cell_starts[0]; cell_index[0] < cell_ends[0];
                   cell_index[0]++) {

                // convert the cartesian cell index into a mesh heirarchy
                // index
                INT mh_tuple[6];
                mesh_hierarchy_device_mapper.cart_tuple_to_tuple(cell_index,
                                                                 mh_tuple);
                // convert the mesh hierarchy tuple to linear index
                const INT linear_index =
                    mesh_hierarchy_device_mapper.tuple_to_linear_global(
                        mh_tuple);

                // now we actually have a MeshHierarchy linear index to
                // test for composite geoms
                CompositeCollection *cc;
                const bool cell_exists = k_MAP_ROOT->get(linear_index, &cc);

                if (cell_exists) {
                  const int num_tris = cc->num_tris;

                  REAL xi0, xi1, xi2, eta0, eta1, eta2;
                  for (int gx = 0; gx < num_tris; gx++) {
                    // get the plane of the geom
                    const LinePlaneIntersection *lpi = &cc->lpi_tris[gx];
                    // does the trajectory intersect the plane
                    if (lpi->line_segment_intersection(p00, p01, p02, p10, p11,
                                                       p12, &i0, &i1, &i2)) {
                      // is the intersection point near to the geom
                      if (lpi->point_near_to_geom(i0, i1, i2)) {

                        const Newton::MappingTriangleLinear2DEmbed3D::DataDevice
                            *map_data = cc->buf_tris + gx;
                        Newton::MappingNewtonIterationBase<
                            Newton::MappingTriangleLinear2DEmbed3D>
                            k_newton_type{};
                        Newton::XMapNewtonKernel<
                            Newton::MappingTriangleLinear2DEmbed3D>
                            k_newton_kernel;
                        const bool converged = k_newton_kernel.x_inverse(
                            map_data, i0, i1, i2, &xi0, &xi1, &xi2, nullptr,
                            k_max_iterations, k_newton_tol);

                        k_newton_type.loc_coord_to_loc_collapsed(
                            map_data, xi0, xi1, xi2, &eta0, &eta1, &eta2);

                        eta0 = Kernel::min(eta0, 1.0 + k_contained_tol);
                        eta1 = Kernel::min(eta1, 1.0 + k_contained_tol);
                        eta2 = Kernel::min(eta2, 1.0 + k_contained_tol);
                        eta0 = Kernel::max(eta0, -1.0 - k_contained_tol);
                        eta1 = Kernel::max(eta1, -1.0 - k_contained_tol);
                        eta2 = Kernel::max(eta2, -1.0 - k_contained_tol);

                        k_newton_type.loc_collapsed_to_loc_coord(
                            map_data, eta0, eta1, eta2, &xi0, &xi1, &xi2);

                        const REAL clamped_residual =
                            k_newton_type.newton_residual(
                                map_data, xi0, xi1, xi2, i0, i1, i2, &eta0,
                                &eta1, &eta2, nullptr);

                        const bool contained = clamped_residual <= k_newton_tol;

                        if (contained && converged) {
                          const REAL r0 = p00 - i0;
                          const REAL r1 = p01 - i1;
                          const REAL r2 = p02 - i2;
                          const REAL d2 = r0 * r0 + r1 * r1 + r2 * r2;
                          if (d2 < intersection_distance) {
                            k_OUT_P.at(0) = i0;
                            k_OUT_P.at(1) = i1;
                            k_OUT_P.at(2) = i2;
                            k_OUT_C.at(0) = cc->group_ids_tris[gx];
                            k_OUT_C.at(1) = cc->composite_ids_tris[gx];
                            k_OUT_C.at(2) = cc->geom_ids_tris[gx];
                            intersection_distance = d2;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        },
        Access::read(position_dat->sym), Access::read(previous_position_sym),
        Access::write(dat_positions->sym), Access::write(dat_composite->sym))
        ->execute();
  }
}

void CompositeIntersection::free() { this->composite_collections->free(); }

CompositeIntersection::CompositeIntersection(
    SYCLTargetSharedPtr sycl_target,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::map<int, std::vector<int>> boundary_groups,
    ParameterStoreSharedPtr config)
    : sycl_target(sycl_target),
      particle_mesh_interface(particle_mesh_interface),
      ndim(particle_mesh_interface->graph->GetMeshDimension()),
      boundary_groups(boundary_groups),
      num_cells(particle_mesh_interface->get_cell_count()) {

  for (auto pair : boundary_groups) {
    NESOASSERT(pair.first != this->mask,
               "Cannot have a boundary group with label " +
                   std::to_string(this->mask) + ".");
  }

  this->composite_collections = std::make_shared<CompositeCollections>(
      sycl_target, particle_mesh_interface, boundary_groups);
  this->mesh_hierarchy_mapper = std::make_unique<MeshHierarchyMapper>(
      sycl_target, this->particle_mesh_interface->get_mesh_hierarchy());

  this->d_cell_min_maxes = std::make_unique<BufferDevice<int>>(
      this->sycl_target, 2 * this->ndim * this->num_cells);

  this->dh_max_bounding_box_size =
      std::make_unique<BufferDeviceHost<int>>(this->sycl_target, 1);
  this->dh_mh_cells =
      std::make_unique<BufferDeviceHost<INT>>(this->sycl_target, 128);
  this->dh_mh_cells_index =
      std::make_unique<BufferDeviceHost<int>>(this->sycl_target, 1);

  this->line_intersection_tol =
      config->get<REAL>("CompositeIntersection/line_intersection_tol", 1.0e-8);
  this->newton_tol =
      config->get<REAL>("CompositeIntersection/newton_tol", 1.0e-8);
  this->newton_max_iteration =
      config->get<INT>("CompositeIntersection/newton_max_iteration", 51);
  this->contained_tol = config->get<REAL>("CompositeIntersection/contained_tol",
                                          this->newton_tol);
  this->num_modes_factor =
      config->get<REAL>("CompositeIntersection/num_modes_factor", 1);
}

template <typename T>
void CompositeIntersection::pre_integration(std::shared_ptr<T> iteration_set,
                                            Sym<INT> output_sym_composite) {
  this->check_iteration_set(iteration_set);
  auto particle_group = this->get_particle_group(iteration_set);
  const auto position_dat = particle_group->position_dat;
  const int ndim = position_dat->ncomp;
  NESOASSERT(ndim == this->ndim,
             "missmatch between particle ndim and class ndim");
  NESOASSERT(this->sycl_target == particle_group->sycl_target,
             "missmatch of sycl target");

  if (!particle_group->contains_dat(output_sym_composite)) {
    particle_group->add_particle_dat(
        ParticleDat(this->sycl_target, ParticleProp(output_sym_composite, 3),
                    particle_group->domain->mesh->get_cell_count()));
  }
  NESOASSERT(particle_group->get_dat(output_sym_composite)->ncomp > 2,
             "Insufficent components for output_sym_composite.");

  // If the previous position dat does not already exist create it here
  if (!particle_group->contains_dat(previous_position_sym)) {
    particle_group->add_particle_dat(ParticleDat(
        this->sycl_target, ParticleProp(previous_position_sym, ndim),
        particle_group->domain->mesh->get_cell_count()));
  }

  // copy the current position onto the previous position
  particle_loop(
      "CompositeIntersection::pre_integration", iteration_set,
      [=](auto P, auto PP) {
        for (int dimx = 0; dimx < ndim; dimx++) {
          PP.at(dimx) = P.at(dimx);
        }
      },
      Access::read(position_dat->sym), Access::write(previous_position_sym))
      ->execute();
}

template <typename T>
void CompositeIntersection::execute(std::shared_ptr<T> iteration_set,
                                    Sym<INT> output_sym_composite,
                                    Sym<REAL> output_sym_position) {

  this->check_iteration_set(iteration_set);
  auto particle_group = this->get_particle_group(iteration_set);

  NESOASSERT(
      particle_group->contains_dat(previous_position_sym),
      "Previous position ParticleDat not found. Was pre_integration called?");

  if (!particle_group->contains_dat(output_sym_composite)) {
    particle_group->add_particle_dat(
        ParticleDat(this->sycl_target, ParticleProp(output_sym_composite, 3),
                    particle_group->domain->mesh->get_cell_count()));
  }
  if (!particle_group->contains_dat(output_sym_position)) {
    const int ncomp = particle_group->position_dat->ncomp;
    particle_group->add_particle_dat(
        ParticleDat(this->sycl_target, ParticleProp(output_sym_position, ncomp),
                    particle_group->domain->mesh->get_cell_count()));
  }

  ParticleDatSharedPtr<REAL> dat_positions =
      particle_group->get_dat(output_sym_position);
  NESOASSERT(dat_positions->ncomp >= this->ndim,
             "Insuffient number of components.");
  ParticleDatSharedPtr<INT> dat_composite =
      particle_group->get_dat(output_sym_composite);
  NESOASSERT(dat_composite->ncomp >= 3, "Insuffient number of components.");

  // find the MeshHierarchy cells that the particles potentially pass though
  std::set<INT> mh_cells;
  this->find_cells(iteration_set, mh_cells);

  // Collect the geometry objects for the composites of interest for these
  // cells. On exit from this function mh_cells contains only the new mesh
  // hierarchy cells which were collected.
  this->composite_collections->collect_geometry(mh_cells);

  const auto k_ndim = particle_group->position_dat->ncomp;
  const auto k_mask = this->mask;
  particle_loop(
      "CompositeIntersection::execute_init", iteration_set,
      [=](auto C, auto P) {
        for (int dimx = 0; dimx < k_ndim; dimx++) {
          P.at(dimx) = 0;
        }
        C.at(0) = k_mask;
        C.at(1) = 0;
        C.at(2) = 0;
      },
      Access::write(dat_composite->sym), Access::write(dat_positions->sym))
      ->execute();

  // find the intersection points for the composites
  if (this->ndim == 3) {
    this->find_intersections_3d(iteration_set, dat_composite, dat_positions);
  } else {
    this->find_intersections_2d(iteration_set, dat_composite, dat_positions);
  }
}

template <typename T>
std::map<int, ParticleSubGroupSharedPtr>
CompositeIntersection::get_intersections(std::shared_ptr<T> iteration_set,
                                         Sym<INT> output_sym_composite,
                                         Sym<REAL> output_sym_position) {

  // Get the intersections with composites
  this->execute(iteration_set, output_sym_composite, output_sym_position);

  // Collect the intersections into ParticleSubGroups
  const auto k_mask = this->mask;
  auto particle_hitting_composites = static_particle_sub_group(
      iteration_set,
      [=](auto C) {
        // If the first component is set then the particle hit a composite.
        return C.at(0) != k_mask;
      },
      Access::read(output_sym_composite));

  // split into ParticleSubGroups per composite hit
  std::map<int, ParticleSubGroupSharedPtr> map_composites_to_particles;
  for (const auto &pair : this->boundary_groups) {
    const auto k_boundary_label = pair.first;
    map_composites_to_particles[k_boundary_label] = static_particle_sub_group(
        particle_hitting_composites,
        [=](auto C) { return C.at(0) == k_boundary_label; },
        Access::read(output_sym_composite));
  }

  return map_composites_to_particles;
}

template void
CompositeIntersection::find_cells(std::shared_ptr<ParticleGroup> iteration_set,
                                  std::set<INT> &cells);

template void CompositeIntersection::find_intersections_3d(
    std::shared_ptr<ParticleGroup> iteration_set,
    ParticleDatSharedPtr<INT> dat_composite,
    ParticleDatSharedPtr<REAL> dat_positions);

template void CompositeIntersection::pre_integration(
    std::shared_ptr<ParticleGroup> iteration_set,
    Sym<INT> output_sym_composite =
        Sym<INT>(CompositeIntersection::output_sym_composite_name));

template void CompositeIntersection::execute(
    std::shared_ptr<ParticleGroup> iteration_set,
    Sym<INT> output_sym_composite =
        Sym<INT>(CompositeIntersection::output_sym_composite_name),
    Sym<REAL> output_sym_position =
        Sym<REAL>(CompositeIntersection::output_sym_position_name));

template std::map<int, ParticleSubGroupSharedPtr>
CompositeIntersection::get_intersections(
    std::shared_ptr<ParticleGroup> iteration_set,
    Sym<INT> output_sym_composite =
        Sym<INT>(CompositeIntersection::output_sym_composite_name),
    Sym<REAL> output_sym_position =
        Sym<REAL>(CompositeIntersection::output_sym_position_name));

template void CompositeIntersection::find_cells(
    std::shared_ptr<ParticleSubGroup> iteration_set, std::set<INT> &cells);

template void CompositeIntersection::find_intersections_3d(
    std::shared_ptr<ParticleSubGroup> iteration_set,
    ParticleDatSharedPtr<INT> dat_composite,
    ParticleDatSharedPtr<REAL> dat_positions);

template void CompositeIntersection::pre_integration(
    std::shared_ptr<ParticleSubGroup> iteration_set,
    Sym<INT> output_sym_composite =
        Sym<INT>(CompositeIntersection::output_sym_composite_name));

template void CompositeIntersection::execute(
    std::shared_ptr<ParticleSubGroup> iteration_set,
    Sym<INT> output_sym_composite =
        Sym<INT>(CompositeIntersection::output_sym_composite_name),
    Sym<REAL> output_sym_position =
        Sym<REAL>(CompositeIntersection::output_sym_position_name));

template std::map<int, ParticleSubGroupSharedPtr>
CompositeIntersection::get_intersections(
    std::shared_ptr<ParticleSubGroup> iteration_set,
    Sym<INT> output_sym_composite =
        Sym<INT>(CompositeIntersection::output_sym_composite_name),
    Sym<REAL> output_sym_position =
        Sym<REAL>(CompositeIntersection::output_sym_position_name));

} // namespace NESO::CompositeInteraction
