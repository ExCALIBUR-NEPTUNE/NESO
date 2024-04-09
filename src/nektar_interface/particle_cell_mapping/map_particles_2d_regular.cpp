#include <nektar_interface/particle_cell_mapping/map_particles_2d_regular.hpp>

namespace NESO {

MapParticles2DRegular::MapParticles2DRegular(
    SYCLTargetSharedPtr sycl_target,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    ParameterStoreSharedPtr config)
    : CoarseMappersBase(sycl_target),
      particle_mesh_interface(particle_mesh_interface) {

  this->tol = config->get<REAL>("MapParticles2DRegular/tol", 0.0);

  // filter out the non-regular elements
  // process locally owned elements
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
      geoms_local;
  {
    // Get the locally owned elements
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
        geoms_local_tmp;
    get_all_elements_2d(particle_mesh_interface->graph, geoms_local_tmp);
    for (auto &geom : geoms_local_tmp) {
      if (geom.second->GetMetricInfo()->GetGtype() == eRegular) {
        geoms_local[geom.first] = geom.second;
      }
    }
    geoms_local_tmp.clear();
  }

  // process remote (halo) elements
  std::vector<std::shared_ptr<RemoteGeom2D<Geometry2D>>> geoms_remote;
  {
    std::vector<std::shared_ptr<RemoteGeom2D<Geometry2D>>> geoms_remote_tmp;
    combine_remote_geoms_2d(particle_mesh_interface->remote_triangles,
                            geoms_remote_tmp);
    combine_remote_geoms_2d(particle_mesh_interface->remote_quads,
                            geoms_remote_tmp);
    geoms_remote.reserve(geoms_remote_tmp.size());
    for (auto &geom : geoms_remote_tmp) {
      if (geom->geom->GetMetricInfo()->GetGtype() == eRegular) {
        geoms_remote.push_back(geom);
      }
    }
    geoms_remote_tmp.clear();
  }

  const int cell_count = geoms_local.size() + geoms_remote.size();
  this->num_regular_geoms = cell_count;

  if (this->num_regular_geoms > 0) {

    // create the coarse lookup mesh
    this->coarse_lookup_map = std::make_unique<CoarseLookupMap>(
        2, this->sycl_target, geoms_local, geoms_remote);

    // store the information required to evaluate v_GetLocCoords for regular
    // Geometry2D objects.
    // map from cartesian cells to nektar mesh cells
    std::map<int, std::list<std::pair<double, int>>> geom_map;
    this->dh_cell_ids =
        std::make_unique<BufferDeviceHost<int>>(this->sycl_target, cell_count);
    this->dh_mpi_ranks =
        std::make_unique<BufferDeviceHost<int>>(this->sycl_target, cell_count);
    this->dh_type =
        std::make_unique<BufferDeviceHost<int>>(this->sycl_target, cell_count);

    this->dh_vertices = std::make_unique<BufferDeviceHost<double>>(
        this->sycl_target, cell_count * 6);

    const int index_tri_geom =
        shape_type_to_int(LibUtilities::ShapeType::eTriangle);
    const int index_quad_geom =
        shape_type_to_int(LibUtilities::ShapeType::eQuadrilateral);
    const int rank = this->sycl_target->comm_pair.rank_parent;

    for (auto &geom : geoms_local) {
      const int id = geom.second->GetGlobalID();
      const int cell_index = this->coarse_lookup_map->gid_to_lookup_id.at(id);
      NESOASSERT((cell_index < cell_count) && (0 <= cell_index),
                 "Bad cell index from map.");
      NESOASSERT(id == geom.first, "ID mismatch");
      this->dh_cell_ids->h_buffer.ptr[cell_index] = id;
      this->dh_mpi_ranks->h_buffer.ptr[cell_index] = rank;
      const int geom_type = shape_type_to_int(geom.second->GetShapeType());
      NESOASSERT((geom_type == index_tri_geom) ||
                     (geom_type == index_quad_geom),
                 "Unknown shape type.");
      this->dh_type->h_buffer.ptr[cell_index] = geom_type;
      this->write_vertices_2d(geom.second, cell_index,
                              this->dh_vertices->h_buffer.ptr);
    }
    for (auto &geom : geoms_remote) {
      const int id = geom->id;
      const int cell_index = this->coarse_lookup_map->gid_to_lookup_id.at(id);
      NESOASSERT((cell_index < cell_count) && (0 <= cell_index),
                 "Bad cell index from map.");
      this->dh_cell_ids->h_buffer.ptr[cell_index] = id;
      this->dh_mpi_ranks->h_buffer.ptr[cell_index] = geom->rank;
      const int geom_type = shape_type_to_int(geom->geom->GetShapeType());
      NESOASSERT((geom_type == index_tri_geom) ||
                     (geom_type == index_quad_geom),
                 "Unknown shape type.");
      this->dh_type->h_buffer.ptr[cell_index] = geom_type;
      this->write_vertices_2d(geom->geom, cell_index,
                              this->dh_vertices->h_buffer.ptr);
    }
    this->dh_cell_ids->host_to_device();
    this->dh_mpi_ranks->host_to_device();
    this->dh_type->host_to_device();
    this->dh_vertices->host_to_device();
  }
}

void MapParticles2DRegular::map(ParticleGroup &particle_group,
                                const int map_cell) {

  // This method will only map into regular geoms (triangles and quads which
  // are parallelograms).
  if (this->num_regular_geoms == 0) {
    return;
  }

  auto &clm = this->coarse_lookup_map;

  // Get kernel pointers to the mesh data.
  const auto &mesh = clm->cartesian_mesh;
  const auto k_mesh_cell_count = mesh->get_cell_count();
  const auto k_mesh_origin0 = mesh->dh_origin->h_buffer.ptr[0];
  const auto k_mesh_origin1 = mesh->dh_origin->h_buffer.ptr[1];
  const auto k_mesh_cell_counts0 = mesh->dh_cell_counts->h_buffer.ptr[0];
  const auto k_mesh_cell_counts1 = mesh->dh_cell_counts->h_buffer.ptr[1];
  const auto k_mesh_inverse_cell_widths0 =
      mesh->dh_inverse_cell_widths->h_buffer.ptr[0];
  const auto k_mesh_inverse_cell_widths1 =
      mesh->dh_inverse_cell_widths->h_buffer.ptr[1];

  // Get kernel pointers to the map data.
  const auto k_map_cell_ids = this->dh_cell_ids->d_buffer.ptr;
  const auto k_map_mpi_ranks = this->dh_mpi_ranks->d_buffer.ptr;
  const auto k_map_type = this->dh_type->d_buffer.ptr;
  const auto k_map_vertices = this->dh_vertices->d_buffer.ptr;
  const auto k_map = clm->dh_map->d_buffer.ptr;
  const auto k_map_sizes = clm->dh_map_sizes->d_buffer.ptr;
  const auto k_map_stride = clm->map_stride;
  const int k_geom_is_triangle =
      shape_type_to_int(LibUtilities::ShapeType::eTriangle);
  const double k_tol = this->tol;

  // Get kernel pointers to the ParticleDats
  const auto position_dat = particle_group.position_dat;
  const auto k_part_positions = position_dat->cell_dat.device_ptr();
  auto k_part_cell_ids = particle_group.cell_id_dat->cell_dat.device_ptr();
  auto k_part_mpi_ranks = particle_group.mpi_rank_dat->cell_dat.device_ptr();
  auto k_part_ref_positions =
      particle_group[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
          ->cell_dat.device_ptr();

  // Get iteration set for particles, two cases single cell case or all cells
  const int max_cell_occupancy = (map_cell > -1)
                                     ? position_dat->h_npart_cell[map_cell]
                                     : position_dat->cell_dat.get_nrow_max();
  const int k_cell_offset = (map_cell > -1) ? map_cell : 0;
  const std::size_t local_size = 256;
  const auto div_mod = std::div(max_cell_occupancy, local_size);
  const int outer_size = div_mod.quot + (div_mod.rem == 0 ? 0 : 1);
  const std::size_t cell_count =
      (map_cell > -1) ? 1
                      : static_cast<std::size_t>(position_dat->cell_dat.ncells);
  sycl::range<2> outer_iterset{local_size * outer_size, cell_count};
  sycl::range<2> local_iterset{local_size, 1};
  const auto k_npart_cell = position_dat->d_npart_cell;
  auto k_ep = this->ep->device_ptr();

  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(
            sycl::nd_range<2>(outer_iterset, local_iterset),
            [=](sycl::nd_item<2> idx) {
              const int cellx = idx.get_global_id(1) + k_cell_offset;
              const int layerx = idx.get_global_id(0);
              if (layerx < k_npart_cell[cellx]) {
                if (k_part_mpi_ranks[cellx][1][layerx] < 0) {

                  // read the position of the particle
                  const double p0 = k_part_positions[cellx][0][layerx];
                  const double p1 = k_part_positions[cellx][1][layerx];
                  const double shifted_p0 = p0 - k_mesh_origin0;
                  const double shifted_p1 = p1 - k_mesh_origin1;

                  // determine the cartesian mesh cell for the position
                  int c0 = (k_mesh_inverse_cell_widths0 * shifted_p0);
                  int c1 = (k_mesh_inverse_cell_widths1 * shifted_p1);
                  c0 = (c0 < 0) ? 0 : c0;
                  c1 = (c1 < 0) ? 0 : c1;
                  c0 = (c0 >= k_mesh_cell_counts0) ? k_mesh_cell_counts0 - 1
                                                   : c0;
                  c1 = (c1 >= k_mesh_cell_counts1) ? k_mesh_cell_counts1 - 1
                                                   : c1;
                  const int linear_mesh_cell = c0 + k_mesh_cell_counts0 * c1;

                  const bool valid_cell =
                      (linear_mesh_cell >= 0) &&
                      (linear_mesh_cell < k_mesh_cell_count);

                  const double r0 = p0;
                  const double r1 = p1;

                  bool cell_found = false;
                  for (int candidate_cell = 0;
                       (candidate_cell < k_map_sizes[linear_mesh_cell]) &&
                       (valid_cell);
                       candidate_cell++) {
                    const int geom_map_index =
                        k_map[linear_mesh_cell * k_map_stride + candidate_cell];

                    const double v00 = k_map_vertices[geom_map_index * 6 + 0];
                    const double v01 = k_map_vertices[geom_map_index * 6 + 1];
                    const double v10 = k_map_vertices[geom_map_index * 6 + 2];
                    const double v11 = k_map_vertices[geom_map_index * 6 + 3];
                    const double v20 = k_map_vertices[geom_map_index * 6 + 4];
                    const double v21 = k_map_vertices[geom_map_index * 6 + 5];

                    const double er_0 = r0 - v00;
                    const double er_1 = r1 - v01;
                    const double er_2 = 0.0;

                    const double e10_0 = v10 - v00;
                    const double e10_1 = v11 - v01;
                    const double e10_2 = 0.0;

                    const double e20_0 = v20 - v00;
                    const double e20_1 = v21 - v01;
                    const double e20_2 = 0.0;

                    MAPPING_CROSS_PRODUCT_3D(e10_0, e10_1, e10_2, e20_0, e20_1,
                                             e20_2, const double norm_0,
                                             const double norm_1,
                                             const double norm_2)
                    MAPPING_CROSS_PRODUCT_3D(norm_0, norm_1, norm_2, e10_0,
                                             e10_1, e10_2, const double orth1_0,
                                             const double orth1_1,
                                             const double orth1_2)
                    MAPPING_CROSS_PRODUCT_3D(norm_0, norm_1, norm_2, e20_0,
                                             e20_1, e20_2, const double orth2_0,
                                             const double orth2_1,
                                             const double orth2_2)

                    const double scale0 =
                        MAPPING_DOT_PRODUCT_3D(er_0, er_1, er_2, orth2_0,
                                               orth2_1, orth2_2) /
                        MAPPING_DOT_PRODUCT_3D(e10_0, e10_1, e10_2, orth2_0,
                                               orth2_1, orth2_2);
                    const double xi0 = 2.0 * scale0 - 1.0;
                    const double scale1 =
                        MAPPING_DOT_PRODUCT_3D(er_0, er_1, er_2, orth1_0,
                                               orth1_1, orth1_2) /
                        MAPPING_DOT_PRODUCT_3D(e20_0, e20_1, e20_2, orth1_0,
                                               orth1_1, orth1_2);
                    const double xi1 = 2.0 * scale1 - 1.0;

                    const int geom_type = k_map_type[geom_map_index];

                    double tmp_eta0;
                    if (geom_type == k_geom_is_triangle) {
                      NekDouble d1 = 1. - xi1;
                      if (fabs(d1) < NekConstants::kNekZeroTol) {
                        if (d1 >= 0.) {
                          d1 = NekConstants::kNekZeroTol;
                        } else {
                          d1 = -NekConstants::kNekZeroTol;
                        }
                      }
                      tmp_eta0 = 2. * (1. + xi0) / d1 - 1.0;
                    } else {
                      tmp_eta0 = xi0;
                    }
                    const double eta0 = tmp_eta0;
                    const double eta1 = xi1;

                    double dist = 0.0;
                    bool contained = ((eta0 <= 1.0) && (eta0 >= -1.0) &&
                                      (eta1 <= 1.0) && (eta1 >= -1.0));
                    if (!contained) {
                      dist = (eta0 < -1.0) ? (-1.0 - eta0) : 0.0;
                      dist = std::max(dist, (eta0 > 1.0) ? (eta0 - 1.0) : 0.0);
                      dist =
                          std::max(dist, (eta1 < -1.0) ? (-1.0 - eta1) : 0.0);
                      dist = std::max(dist, (eta1 > 1.0) ? (eta1 - 1.0) : 0.0);
                    }

                    cell_found = dist <= k_tol;
                    if (cell_found) {
                      const int geom_id = k_map_cell_ids[geom_map_index];
                      const int mpi_rank = k_map_mpi_ranks[geom_map_index];
                      k_part_cell_ids[cellx][0][layerx] = geom_id;
                      k_part_mpi_ranks[cellx][1][layerx] = mpi_rank;
                      k_part_ref_positions[cellx][0][layerx] = xi0;
                      k_part_ref_positions[cellx][1][layerx] = xi1;
                      break;
                    }
                  }
                }
              }
            });
      })
      .wait_and_throw();
}

} // namespace NESO
