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

  const auto position_dat = get_particle_group(iteration_set)->position_dat;
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
              const int trunc =
                  KERNEL_MAX(0, static_cast<int>(cell_cart[dimx]));
              atomic_fetch_min(&k_cell_min_maxes[indexing_cell_min(
                                   cellx, k_num_cells, dimx, k_ndim)],
                               trunc);
            }
            {
              const INT max_possible_cell =
                  mesh_hierarchy_device_mapper.dims[dimx] *
                  mesh_hierarchy_device_mapper.ncells_dim_fine;
              const int trunc = KERNEL_MIN(max_possible_cell - 1,
                                           static_cast<int>(cell_cart[dimx]));

              atomic_fetch_max(&k_cell_min_maxes[indexing_cell_max(
                                   cellx, k_num_cells, dimx, k_ndim)],
                               trunc);
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
            atomic_fetch_max(k_max_ptr, volume);
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
    std::shared_ptr<T> iteration_set, REAL *d_real, INT *d_int) {
  this->check_iteration_set(iteration_set);
  NESOASSERT(this->ndim == 2, "Method assumes 2 spatial dimensions.");

  const auto position_dat = get_particle_group(iteration_set)->position_dat;
  const int k_ndim = this->ndim;
  const auto mesh_hierarchy_device_mapper =
      this->mesh_hierarchy_mapper->get_device_mapper();

  const REAL k_REAL_MAX = std::numeric_limits<REAL>::max();
  const auto npart_local = get_particle_group(iteration_set)->get_npart_local();

  // the binary map containing the geometry information
  auto k_MAP_ROOT = this->composite_collections->map_cells_collections->root;
  if (k_MAP_ROOT != nullptr) {

    const double k_tol = this->line_intersection_tol;
    const int k_max_iterations = this->newton_max_iteration;

    particle_loop(
        "CompositeIntersection::find_intersections_2d", iteration_set,
        [=](auto INDEX, auto k_P, auto k_PP) {
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
          bool intersection_found = false;
          const auto particle_index = INDEX.get_local_linear_index();
          REAL r0_write = 0.0;
          REAL r1_write = 0.0;
          REAL xi_write = 0.0;
          INT group_id = 0;
          INT geom_id = 0;

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
                      intersection_found = true;
                      intersection_distance = d2;
                      r0_write = i0;
                      r1_write = i1;
                      xi_write =
                          cc->lli_segments[sx].get_reference_coordinate(i0, i1);
                      group_id = cc->group_ids_segments[sx];
                      geom_id = cc->geom_ids_segments[sx];
                    }
                  }
                }
              }
            }
          }

          d_int[particle_index] = intersection_found ? 1 : 0;
          if (intersection_found) {
            d_int[npart_local + particle_index] = group_id;
            d_int[npart_local * 2 + particle_index] = geom_id;
            d_int[npart_local * 3 + particle_index] =
                static_cast<int>(LibUtilities::eSegment);
            d_real[particle_index] = r0_write;
            d_real[npart_local + particle_index] = r1_write;
            d_real[npart_local * 2 + particle_index] = xi_write;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(position_dat->sym),
        Access::read(previous_position_sym))
        ->execute();
  }
}

template <typename T>
void CompositeIntersection::find_intersections_3d(
    std::shared_ptr<T> iteration_set, REAL *d_real, INT *d_int) {
  this->check_iteration_set(iteration_set);
  NESOASSERT(this->ndim == 3, "Method assumes 3 spatial dimensions.");

  const auto position_dat = get_particle_group(iteration_set)->position_dat;
  const int k_ndim = this->ndim;
  const auto mesh_hierarchy_device_mapper =
      this->mesh_hierarchy_mapper->get_device_mapper();

  const REAL k_REAL_MAX = std::numeric_limits<REAL>::max();
  const auto npart_local = get_particle_group(iteration_set)->get_npart_local();

  // the binary map containing the geometry information
  auto k_MAP_ROOT = this->composite_collections->map_cells_collections->root;
  if (k_MAP_ROOT != nullptr) {

    const double k_newton_tol = this->newton_tol;
    const double k_contained_tol = this->contained_tol;
    const int k_max_iterations = this->newton_max_iteration;
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
        [=](auto INDEX, auto k_P, auto k_PP) {
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
          bool intersection_found = false;
          const auto particle_index = INDEX.get_local_linear_index();
          REAL r0_write = 0.0;
          REAL r1_write = 0.0;
          REAL r2_write = 0.0;
          REAL xi0_write = 0.0;
          REAL xi1_write = 0.0;
          INT group_id = 0;
          INT geom_id = 0;

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
                                intersection_found = true;
                                intersection_distance = d2;
                                r0_write = i0;
                                r1_write = i1;
                                r2_write = i2;
                                xi0_write = xi[0];
                                xi1_write = xi[1];
                                group_id = cc->group_ids_quads[gx];
                                geom_id = cc->geom_ids_quads[gx];
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
          if (intersection_found) {
            d_int[particle_index] = 1;
            d_int[npart_local + particle_index] = group_id;
            d_int[npart_local * 2 + particle_index] = geom_id;
            d_int[npart_local * 3 + particle_index] =
                static_cast<int>(LibUtilities::eQuadrilateral);
            d_real[particle_index] = r0_write;
            d_real[npart_local + particle_index] = r1_write;
            d_real[npart_local * 2 + particle_index] = r2_write;
            d_real[npart_local * 3 + particle_index] = xi0_write;
            d_real[npart_local * 4 + particle_index] = xi1_write;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(position_dat->sym),
        Access::read(previous_position_sym))
        ->execute();

    static_assert(!Newton::local_memory_required<
                      Newton::MappingTriangleLinear2DEmbed3D>::required,
                  "Did not expect local memory to be required for this Newton "
                  "implemenation");

    particle_loop(
        "CompositeIntersection::find_intersections_3d_triangles", iteration_set,
        [=](auto INDEX, auto k_P, auto k_PP) {
          const auto particle_index = INDEX.get_local_linear_index();
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

          const bool intersection_already_found = d_int[particle_index];
          const REAL existing_r0 = d_real[particle_index];
          const REAL existing_r1 = d_real[npart_local + particle_index];
          const REAL existing_r2 = d_real[npart_local * 2 + particle_index];

          const REAL d0 = existing_r0 - k_PP.at(0);
          const REAL d1 = existing_r1 - k_PP.at(1);
          const REAL d2 = existing_r2 - k_PP.at(2);
          const REAL intersection_distance_p = d0 * d0 + d1 * d1 + d2 * d2;

          REAL intersection_distance =
              intersection_already_found ? intersection_distance_p : k_REAL_MAX;

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
          bool intersection_found = false;
          REAL r0_write = 0.0;
          REAL r1_write = 0.0;
          REAL r2_write = 0.0;
          REAL xi0_write = 0.0;
          REAL xi1_write = 0.0;
          INT group_id = 0;
          INT geom_id = 0;

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
                            intersection_found = true;
                            intersection_distance = d2;
                            r0_write = i0;
                            r1_write = i1;
                            r2_write = i2;
                            xi0_write = xi0;
                            xi1_write = xi1;
                            group_id = cc->group_ids_tris[gx];
                            geom_id = cc->geom_ids_tris[gx];
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          if (intersection_found) {
            d_int[particle_index] = 1;
            d_int[npart_local + particle_index] = group_id;
            d_int[npart_local * 2 + particle_index] = geom_id;
            d_int[npart_local * 3 + particle_index] =
                static_cast<int>(LibUtilities::eTriangle);
            d_real[particle_index] = r0_write;
            d_real[npart_local + particle_index] = r1_write;
            d_real[npart_local * 2 + particle_index] = r2_write;
            d_real[npart_local * 3 + particle_index] = xi0_write;
            d_real[npart_local * 4 + particle_index] = xi1_write;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(position_dat->sym),
        Access::read(previous_position_sym))
        ->execute();
  }
}

void CompositeIntersection::free() {
  this->composite_collections->free();
  for (const auto &gx : this->map_groups_boundary_interface) {
    gx.second->free();
  }
  this->map_groups_boundary_interface.clear();
}

CompositeIntersection::CompositeIntersection(
    SYCLTargetSharedPtr sycl_target,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::map<int, std::vector<int>> boundary_groups,
    MultiRegions::DisContFieldSharedPtr prototype_field,
    ParameterStoreSharedPtr config)
    : sycl_target(sycl_target),
      particle_mesh_interface(particle_mesh_interface),
      ndim(particle_mesh_interface->graph->GetMeshDimension()),
      boundary_groups(boundary_groups), prototype_field(prototype_field),
      num_cells(particle_mesh_interface->get_cell_count()) {

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

  if (this->prototype_field) {
    this->composite_function_context =
        std::make_shared<CompositeFunctionContext>(
            this->sycl_target, this->particle_mesh_interface->graph,
            this->prototype_field, this->boundary_groups);
    for (auto gx : boundary_groups) {
      this->map_groups_unseen_value_extractor[gx.first] =
          std::make_shared<UnseenValueExtractor>(this->sycl_target);

      this->map_groups_boundary_interface[gx.first] =
          std::make_shared<BoundaryMeshInterface>(
              this->sycl_target->comm_pair.comm_parent, this->sycl_target,
              this->composite_function_context->get_owned_geoms(gx.first));
    }
  }
}

CompositeIntersection::CompositeIntersection(
    SYCLTargetSharedPtr sycl_target,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    std::map<int, std::vector<int>> boundary_groups,
    ParameterStoreSharedPtr config)
    :

      CompositeIntersection(sycl_target, particle_mesh_interface,
                            boundary_groups, nullptr, config) {}

template <typename T>
void CompositeIntersection::pre_integration(std::shared_ptr<T> iteration_set) {
  this->check_iteration_set(iteration_set);
  auto particle_group = get_particle_group(iteration_set);
  const auto position_dat = particle_group->position_dat;
  const int ndim = position_dat->ncomp;
  NESOASSERT(ndim == this->ndim,
             "missmatch between particle ndim and class ndim");
  NESOASSERT(this->sycl_target == particle_group->sycl_target,
             "missmatch of sycl target");

  // If the previous position dat does not already exist create it here
  if (!particle_group->contains_dat(previous_position_sym, this->ndim)) {
    particle_group->add_particle_dat(previous_position_sym, ndim);
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
std::map<int, ParticleSubGroupSharedPtr>
CompositeIntersection::get_intersections(std::shared_ptr<T> iteration_set) {

  this->check_iteration_set(iteration_set);
  auto particle_group = get_particle_group(iteration_set);

  NESOASSERT(
      particle_group->contains_dat(previous_position_sym),
      "Previous position ParticleDat not found. Was pre_integration called?");

  // find the MeshHierarchy cells that the particles potentially pass though
  std::set<INT> mh_cells;
  this->find_cells(iteration_set, mh_cells);

  // Collect the geometry objects for the composites of interest for these
  // cells. On exit from this function mh_cells contains only the new mesh
  // hierarchy cells which were collected.
  this->composite_collections->collect_geometry(mh_cells);

  const auto npart_local = get_particle_group(iteration_set)->get_npart_local();
  auto d_real = get_resource<BufferDevice<REAL>,
                             ResourceStackInterfaceBufferDevice<REAL>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);

  // We store the intersection point and the reference coordinates in the cell
  // of that point.
  d_real->realloc_no_copy(npart_local * (this->ndim + this->ndim - 1));
  REAL *k_real = d_real->ptr;
  auto d_int =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_int->realloc_no_copy(npart_local * 4);
  INT *k_int = d_int->ptr;
  this->sycl_target->queue.fill(k_int, (INT)0, npart_local).wait_and_throw();

  // find the intersection points for the composites
  if (this->ndim == 3) {
    this->find_intersections_3d(iteration_set, k_real, k_int);
  } else {
    this->find_intersections_2d(iteration_set, k_real, k_int);
  }

  // Collect the intersections into ParticleSubGroups
  auto particle_hitting_composites = static_particle_sub_group(
      iteration_set,
      [=](auto INDEX) { return k_int[INDEX.get_local_linear_index()] == 1; },
      Access::read(ParticleLoopIndex{}));

  // split into ParticleSubGroups per composite hit
  std::map<int, ParticleSubGroupSharedPtr> map_composites_to_particles;
  for (const auto &pair : this->boundary_groups) {
    const auto k_boundary_label = pair.first;
    map_composites_to_particles[k_boundary_label] = static_particle_sub_group(
        particle_hitting_composites,
        [=](auto INDEX) {
          return k_int[npart_local + INDEX.get_local_linear_index()] ==
                 k_boundary_label;
        },
        Access::read(ParticleLoopIndex{}));

    add_boundary_interaction_ephemeral_dats(
        map_composites_to_particles[k_boundary_label], this->ndim);
    map_composites_to_particles[k_boundary_label]->add_ephemeral_dat(
        Sym<REAL>("NESO_BOUNDARY_REFERENCE_POSITIONS"), this->ndim - 1);
    map_composites_to_particles[k_boundary_label]->add_ephemeral_dat(
        Sym<INT>("NESO_BOUNDARY_ELEMENT_TYPE"), 1);
  }

  const auto k_normal_device_mapper =
      this->composite_collections->get_device_normal_mapper();

  // Assemble the EphemeralDats
  const auto k_ndim = particle_group->position_dat->ncomp;
  for (const auto &pair : this->boundary_groups) {
    const auto group_id = pair.first;

    if (map_composites_to_particles.count(group_id)) {
      particle_loop(
          map_composites_to_particles[group_id],
          [=](auto INDEX, auto INTERSECTION_POINT, auto METADATA,
              auto REF_COORDS, auto ELEMENT_TYPE) {
            const auto particle_index = INDEX.get_local_linear_index();
            for (int dx = 0; dx < k_ndim; dx++) {
              INTERSECTION_POINT.at_ephemeral(dx) =
                  k_real[dx * npart_local + particle_index];
            }
            for (int dx = 0; dx < k_ndim - 1; dx++) {
              REF_COORDS.at_ephemeral(dx) =
                  k_real[(dx + k_ndim) * npart_local + particle_index];
            }
            METADATA.at_ephemeral(0) = k_int[npart_local + particle_index];
            METADATA.at_ephemeral(1) = k_int[npart_local * 2 + particle_index];
            ELEMENT_TYPE.at_ephemeral(0) =
                k_int[npart_local * 3 + particle_index];
          },
          Access::read(ParticleLoopIndex{}),
          Access::write(
              Sym<REAL>("NESO_PARTICLES_BOUNDARY_INTERSECTION_POINT")),
          Access::write(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
          Access::write(Sym<REAL>("NESO_BOUNDARY_REFERENCE_POSITIONS")),
          Access::write(Sym<INT>("NESO_BOUNDARY_ELEMENT_TYPE")))
          ->execute();

      if (k_normal_device_mapper.root) {
        particle_loop(
            map_composites_to_particles[group_id],
            [=](auto INDEX, auto BOUNDARY_NORMAL) {
              const auto particle_index = INDEX.get_local_linear_index();
              const INT geom_id = k_int[npart_local * 2 + particle_index];
              REAL *normal;
              k_normal_device_mapper.get(geom_id, &normal);

              for (int dx = 0; dx < k_ndim; dx++) {
                BOUNDARY_NORMAL.at_ephemeral(dx) = normal[dx];
              }
            },
            Access::read(ParticleLoopIndex{}),
            Access::write(Sym<REAL>("NESO_PARTICLES_BOUNDARY_NORMAL")))
            ->execute();
      }
    }

    if (this->prototype_field) {
      // Does this rank actually have any particles hitting that boundary group?
      std::set<INT> new_geoms;
      if (map_composites_to_particles.count(group_id)) {
        new_geoms =
            this->map_groups_unseen_value_extractor.at(group_id)->extract(
                map_composites_to_particles.at(group_id),
                BoundaryInteractionSpecification::intersection_metadata, 1,
                true);
      }

      std::vector<std::pair<int, INT>> new_potentialy_hit_geoms;
      new_potentialy_hit_geoms.reserve(new_geoms.size());
      for (auto &geomx : new_geoms) {
        const int owning_rank =
            this->composite_collections->composite_transport->get_owning_rank(
                static_cast<int>(geomx));
        new_potentialy_hit_geoms.push_back({owning_rank, geomx});
      }

      this->map_groups_boundary_interface.at(group_id)->extend_exchange_pattern(
          new_potentialy_hit_geoms);
    }
  }

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_int);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<REAL>{}, d_real);

  return map_composites_to_particles;
}

CompositeFunctionSharedPtr
CompositeIntersection::create_function(const int group) {
  NESOASSERT(this->composite_function_context != nullptr,
             "CompositeIntersection instance was not created with a prototype "
             "field to use for creating boundary functions/fields.");
  return this->composite_function_context->create_function(group);
}

void CompositeIntersection::function_evaluate(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CompositeFunctionSharedPtr func) {

  this->composite_function_context->function_evaluate(
      particle_sub_group, sym, component, is_ephemeral, func,
      this->map_groups_boundary_interface.at(func->boundary_group));
}

void CompositeIntersection::function_project_initialise(
    CompositeFunctionSharedPtr func) {

  auto boundary_mesh_interface =
      this->get_boundary_mesh_interface(func->boundary_group);

  this->composite_function_context->function_project_initialise(
      func, boundary_mesh_interface);
}

void CompositeIntersection::function_project_contribute(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CompositeFunctionSharedPtr func) {
  auto boundary_mesh_interface =
      this->get_boundary_mesh_interface(func->boundary_group);

  this->composite_function_context->function_project_contribute(
      particle_sub_group, sym, component, is_ephemeral, func,
      boundary_mesh_interface);
}

void CompositeIntersection::function_project_finalise_reduce(
    CompositeFunctionSharedPtr func) {

  auto boundary_mesh_interface =
      this->get_boundary_mesh_interface(func->boundary_group);

  this->composite_function_context->function_project_finalise_reduce(
      func, boundary_mesh_interface);
}

void CompositeIntersection::function_project_finalise_mass_solve(
    CompositeFunctionSharedPtr func) {

  auto boundary_mesh_interface =
      this->get_boundary_mesh_interface(func->boundary_group);

  this->composite_function_context->function_project_finalise_mass_solve(
      func, boundary_mesh_interface);
}

void CompositeIntersection::function_project_finalise(
    CompositeFunctionSharedPtr func) {

  auto boundary_mesh_interface =
      this->get_boundary_mesh_interface(func->boundary_group);

  this->composite_function_context->function_project_finalise(
      func, boundary_mesh_interface);
}

void CompositeIntersection::function_project(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CompositeFunctionSharedPtr func) {

  auto boundary_mesh_interface =
      this->get_boundary_mesh_interface(func->boundary_group);

  this->composite_function_context->function_project(
      particle_sub_group, sym, component, is_ephemeral, func,
      boundary_mesh_interface);
}

std::shared_ptr<BoundaryMeshInterface>
CompositeIntersection::get_boundary_mesh_interface(const int group) {

  NESOASSERT(
      this->map_groups_boundary_interface.count(group),
      "Passed group does not exist in map from groups to boundary interfaces.");
  return this->map_groups_boundary_interface[group];
}

template void
CompositeIntersection::find_cells(std::shared_ptr<ParticleGroup> iteration_set,
                                  std::set<INT> &cells);

template void CompositeIntersection::find_intersections_2d(
    std::shared_ptr<ParticleGroup> iteration_set, REAL *d_real, INT *d_int);

template void CompositeIntersection::find_intersections_3d(
    std::shared_ptr<ParticleGroup> iteration_set, REAL *d_real, INT *d_int);

template void CompositeIntersection::pre_integration(
    std::shared_ptr<ParticleGroup> iteration_set);

template std::map<int, ParticleSubGroupSharedPtr>
CompositeIntersection::get_intersections(
    std::shared_ptr<ParticleGroup> iteration_set);

template void CompositeIntersection::find_cells(
    std::shared_ptr<ParticleSubGroup> iteration_set, std::set<INT> &cells);

template void CompositeIntersection::find_intersections_2d(
    std::shared_ptr<ParticleSubGroup> iteration_set, REAL *d_real, INT *d_int);

template void CompositeIntersection::find_intersections_3d(
    std::shared_ptr<ParticleSubGroup> iteration_set, REAL *d_real, INT *d_int);

template void CompositeIntersection::pre_integration(
    std::shared_ptr<ParticleSubGroup> iteration_set);

template std::map<int, ParticleSubGroupSharedPtr>
CompositeIntersection::get_intersections(
    std::shared_ptr<ParticleSubGroup> iteration_set);
} // namespace NESO::CompositeInteraction
