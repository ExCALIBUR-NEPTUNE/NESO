#ifndef __PARTICLE_CELL_MAPPING_3D_H__
#define __PARTICLE_CELL_MAPPING_3D_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "candidate_cell_mapping.hpp"
#include "coordinate_mapping.hpp"
#include "geometry_transport_2d.hpp"
#include "geometry_transport_3d.hpp"
#include "particle_cell_mapping_common.hpp"
#include "particle_mesh_interface.hpp"

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

class MapParticles3DRegular {
protected:
  /// Disable (implicit) copies.
  MapParticles3DRegular(const MapParticles3DRegular &st) = delete;
  /// Disable (implicit) copies.
  MapParticles3DRegular &operator=(MapParticles3DRegular const &a) = delete;

  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  std::unique_ptr<CoarseLookupMap> coarse_lookup_map;

  int num_regular_geoms;

  /// The nektar++ cell id for the cells indices pointed to from the map.
  std::unique_ptr<BufferDeviceHost<int>> dh_cell_ids;
  /// The MPI rank that owns the cell.
  std::unique_ptr<BufferDeviceHost<int>> dh_mpi_ranks;
  /// The type of the cell, i.e. a quad or a triangle.
  std::unique_ptr<BufferDeviceHost<int>> dh_type;
  /// The 3 vertices required by mapping from physical space to reference space.
  std::unique_ptr<BufferDeviceHost<double>> dh_vertices;

  std::unique_ptr<ErrorPropagate> ep;

  template <typename U>
  inline void write_vertices_3d(U &geom, const int index, double *output) {
    const auto shape_type = geom->GetShapeType();
    int index_v[4];
    index_v[0] = 0; // v0 is actually 0
    if (shape_type == LibUtilities::eHexahedron ||
        shape_type == LibUtilities::ePrism ||
        shape_type == LibUtilities::ePyramid) {
      index_v[1] = 1;
      index_v[2] = 3;
      index_v[3] = 4;
    } else if (shape_type == LibUtilities::eTetrahedron) {
      index_v[1] = 1;
      index_v[2] = 2;
      index_v[3] = 3;
    } else {
      NESOASSERT(false, "get_local_coords_3d Unknown shape type.");
    }

    for (int vx = 0; vx < 4; vx++) {
      auto vertex = geom->GetVertex(index_v[vx]);
      NESOASSERT(vertex->GetCoordim() == 3, "Expected Coordim to be 3.");
      NekDouble x, y, z;
      vertex->GetCoords(x, y, z);
      output[index * 12 + vx * 3 + 0] = x;
      output[index * 12 + vx * 3 + 1] = y;
      output[index * 12 + vx * 3 + 2] = z;
    }
  }

public:
  MapParticles3DRegular(SYCLTargetSharedPtr sycl_target,
                        ParticleMeshInterfaceSharedPtr particle_mesh_interface)
      : sycl_target(sycl_target),
        particle_mesh_interface(particle_mesh_interface) {

    // filter out the non-regular elements
    // process locally owned elements
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
        geoms_local;
    {
      // Get the locally owned elements
      std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
          geoms_local_tmp;
      get_all_elements_3d(particle_mesh_interface->graph, geoms_local_tmp);
      for (auto &geom : geoms_local_tmp) {
        if (geom.second->GetMetricInfo()->GetGtype() == eRegular) {
          geoms_local[geom.first] = geom.second;
        }
      }
      geoms_local_tmp.clear();
    }

    // process remote (halo) elements
    std::vector<std::shared_ptr<RemoteGeom3D>> geoms_remote;
    {
      auto &geoms_remote_tmp = particle_mesh_interface->remote_geoms_3d;
      for (auto &geom : geoms_remote_tmp) {
        if (geom->geom->GetMetricInfo()->GetGtype() == eRegular) {
          geoms_remote.push_back(geom);
        }
      }
    }

    const int cell_count = geoms_local.size() + geoms_remote.size();
    this->num_regular_geoms = cell_count;
    if (this->num_regular_geoms > 0) {

      // create the coarse lookup mesh
      this->coarse_lookup_map = std::make_unique<CoarseLookupMap>(
          3, this->sycl_target, geoms_local, geoms_remote);

      // store the information required to evaluate v_GetLocCoords for regular
      // Geometry3D objects.
      // map from cartesian cells to nektar mesh cells
      std::map<int, std::list<std::pair<double, int>>> geom_map;
      this->dh_cell_ids = std::make_unique<BufferDeviceHost<int>>(
          this->sycl_target, cell_count);
      this->dh_mpi_ranks = std::make_unique<BufferDeviceHost<int>>(
          this->sycl_target, cell_count);
      this->dh_type = std::make_unique<BufferDeviceHost<int>>(this->sycl_target,
                                                              cell_count);

      this->dh_vertices = std::make_unique<BufferDeviceHost<double>>(
          this->sycl_target, cell_count * 12);

      const int index_tet = shape_type_to_int(eTetrahedron);
      const int index_pyr = shape_type_to_int(ePyramid);
      const int index_prism = shape_type_to_int(ePrism);
      const int index_hex = shape_type_to_int(eHexahedron);

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
        NESOASSERT((geom_type == index_tet) || (geom_type == index_pyr) ||
                       (geom_type == index_prism) || (geom_type == index_hex),
                   "Unknown shape type.");
        this->dh_type->h_buffer.ptr[cell_index] = geom_type;
        this->write_vertices_3d(geom.second, cell_index,
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
        NESOASSERT((geom_type == index_tet) || (geom_type == index_pyr) ||
                       (geom_type == index_prism) || (geom_type == index_hex),
                   "Unknown shape type.");
        this->dh_type->h_buffer.ptr[cell_index] = geom_type;
        this->write_vertices_3d(geom->geom, cell_index,
                                this->dh_vertices->h_buffer.ptr);
      }

      this->dh_cell_ids->host_to_device();
      this->dh_mpi_ranks->host_to_device();
      this->dh_type->host_to_device();
      this->dh_vertices->host_to_device();
    }
    this->ep = std::make_unique<ErrorPropagate>(this->sycl_target);
  }

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  3D Geometry objects.
   */
  inline void map(ParticleGroup &particle_group, const int map_cell = -1,
                  const double tol = 0.0) {

    // This method will only map into regular geoms.
    if (this->num_regular_geoms == 0) {
      return;
    }

    auto &clm = this->coarse_lookup_map;
    // Get kernel pointers to the mesh data.
    const auto &mesh = clm->cartesian_mesh;
    const auto k_mesh_cell_count = mesh->get_cell_count();
    const auto k_mesh_origin = mesh->dh_origin->d_buffer.ptr;
    const auto k_mesh_cell_counts = mesh->dh_cell_counts->d_buffer.ptr;
    const auto k_mesh_inverse_cell_widths =
        mesh->dh_inverse_cell_widths->d_buffer.ptr;
    // Get kernel pointers to the map data.
    const auto k_map_cell_ids = this->dh_cell_ids->d_buffer.ptr;
    const auto k_map_mpi_ranks = this->dh_mpi_ranks->d_buffer.ptr;
    const auto k_map_type = this->dh_type->d_buffer.ptr;
    const auto k_map_vertices = this->dh_vertices->d_buffer.ptr;
    const auto k_map = clm->dh_map->d_buffer.ptr;
    const auto k_map_sizes = clm->dh_map_sizes->d_buffer.ptr;
    const auto k_map_stride = clm->map_stride;
    const double k_tol = tol;

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
    const size_t local_size = 256;
    const auto div_mod = std::div(max_cell_occupancy, local_size);
    const int outer_size = div_mod.quot + (div_mod.rem == 0 ? 0 : 1);
    const size_t cell_count =
        (map_cell > -1) ? 1
                        : static_cast<size_t>(position_dat->cell_dat.ncells);
    sycl::range<2> outer_iterset{local_size * outer_size, cell_count};
    sycl::range<2> local_iterset{local_size, 1};
    const auto k_npart_cell = position_dat->d_npart_cell;

    this->ep->reset();
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
                    const double p2 = k_part_positions[cellx][2][layerx];

                    const double l_pos_tmp[3] = {p0, p1, p2};
                    sycl::private_ptr<const double> p_pos_tmp{l_pos_tmp};
                    sycl::vec<double, 3> v_pos_tmp{};
                    v_pos_tmp.load(0, p_pos_tmp);

                    sycl::global_ptr<const double> p_mesh_origin{k_mesh_origin};
                    sycl::vec<double, 3> v_mesh_origin{};
                    v_mesh_origin.load(0, p_mesh_origin);

                    const auto v_shifted_p = v_pos_tmp - v_mesh_origin;
                    sycl::global_ptr<const double> p_mesh_inverse_cell_widths{
                        k_mesh_inverse_cell_widths};
                    sycl::vec<double, 3> v_mesh_inverse_cell_widths{};
                    v_mesh_inverse_cell_widths.load(0,
                                                    p_mesh_inverse_cell_widths);

                    // Bin into cell as floats
                    const auto v_real_cell =
                        v_shifted_p * v_mesh_inverse_cell_widths;
                    sycl::vec<double, 3> v_trunc_real_cell =
                        sycl::trunc(v_real_cell);
                    double l_trunc_real_cell[3];
                    sycl::private_ptr<double> p_trunc_real_cell{
                        l_trunc_real_cell};
                    v_trunc_real_cell.store(0, p_trunc_real_cell);

                    // Bin into cell as ints
                    int l_trunc_int_cell[3];
                    for (int dimx = 0; dimx < 3; dimx++) {
                      int cx = l_trunc_real_cell[dimx];
                      cx = (cx < 0) ? 0 : cx;
                      const int max_cell = k_mesh_cell_counts[dimx] - 1;
                      cx = (cx > max_cell) ? max_cell : cx;
                      l_trunc_int_cell[dimx] = cx;
                    }

                    // convert to linear index
                    const int c0 = l_trunc_int_cell[0];
                    const int c1 = l_trunc_int_cell[1];
                    const int c2 = l_trunc_int_cell[2];
                    const int mcc0 = k_mesh_cell_counts[0];
                    const int mcc1 = k_mesh_cell_counts[1];
                    const int linear_mesh_cell =
                        c0 + c1 * mcc0 + c2 * mcc0 * mcc1;

                    const bool valid_cell =
                        (linear_mesh_cell >= 0) &&
                        (linear_mesh_cell < k_mesh_cell_count);
                    // loop over the candidate geometry objects
                    bool cell_found = false;
                    for (int candidate_cell = 0;
                         (candidate_cell < k_map_sizes[linear_mesh_cell]) &&
                         (valid_cell);
                         candidate_cell++) {
                      const int geom_map_index =
                          k_map[linear_mesh_cell * k_map_stride +
                                candidate_cell];

                      sycl::global_ptr<const double> p_map_vertices{
                          &k_map_vertices[geom_map_index * 12]};
                      sycl::vec<double, 3> v0{};
                      sycl::vec<double, 3> v1{};
                      sycl::vec<double, 3> v2{};
                      sycl::vec<double, 3> v3{};
                      v0.load(0, p_map_vertices);
                      v1.load(1, p_map_vertices);
                      v2.load(2, p_map_vertices);
                      v3.load(3, p_map_vertices);
                      const auto r = v_pos_tmp;

                      const auto er0 = r - v0;
                      const auto e10 = v1 - v0;
                      const auto e20 = v2 - v0;
                      const auto e30 = v3 - v0;
                      const auto cp1020 = sycl::cross(e10, e20);
                      const auto cp2030 = sycl::cross(e20, e30);
                      const auto cp3010 = sycl::cross(e30, e10);

                      const double iV = 2.0 / sycl::dot(e30, cp1020);
                      double Lcoords[3]; // xi
                      Lcoords[0] = sycl::dot(er0, cp2030) * iV - 1.0;
                      Lcoords[1] = sycl::dot(er0, cp3010) * iV - 1.0;
                      Lcoords[2] = sycl::dot(er0, cp1020) * iV - 1.0;

                      sycl::vec<double, 3> v_xi{0.0};
                      v_xi[0] = Lcoords[0];
                      v_xi[1] = Lcoords[1];
                      v_xi[2] = Lcoords[2];

                      sycl::vec<double, 3> v_eta{0.0};
                      const int geom_type = k_map_type[geom_map_index];
                      GeometryInterface::loc_coord_to_loc_collapsed(
                          geom_type, v_xi, v_eta);

                      const double eta0 = v_eta[0];
                      const double eta1 = v_eta[1];
                      const double eta2 = v_eta[2];
                      bool contained =
                          ((eta0 <= 1.0) && (eta0 >= -1.0) && (eta1 <= 1.0) &&
                           (eta1 >= -1.0) && (eta2 <= 1.0) && (eta2 >= -1.0));

                      double dist = 0.0;
                      if (!contained) {
                        dist = (eta0 < -1.0) ? (-1.0 - eta0) : 0.0;
                        dist =
                            std::max(dist, (eta0 > 1.0) ? (eta0 - 1.0) : 0.0);
                        dist =
                            std::max(dist, (eta1 < -1.0) ? (-1.0 - eta1) : 0.0);
                        dist =
                            std::max(dist, (eta1 > 1.0) ? (eta1 - 1.0) : 0.0);
                        dist =
                            std::max(dist, (eta2 < -1.0) ? (-1.0 - eta2) : 0.0);
                        dist =
                            std::max(dist, (eta2 > 1.0) ? (eta2 - 1.0) : 0.0);
                      }

                      cell_found = dist <= k_tol;
                      if (cell_found) {
                        const int geom_id = k_map_cell_ids[geom_map_index];
                        const int mpi_rank = k_map_mpi_ranks[geom_map_index];
                        k_part_cell_ids[cellx][0][layerx] = geom_id;
                        k_part_mpi_ranks[cellx][1][layerx] = mpi_rank;
                        k_part_ref_positions[cellx][0][layerx] = Lcoords[0];
                        k_part_ref_positions[cellx][1][layerx] = Lcoords[1];
                        k_part_ref_positions[cellx][2][layerx] = Lcoords[2];
                        break;
                      }
                    }
                  }
                }
              });
        })
        .wait_and_throw();
  }
};

/**
 *  Class to map particles into Nektar++ cells. TODO extend to "deformed"
 *  elements to support curved boundaries.
 */
class MapParticles3D {
protected:
  /// Disable (implicit) copies.
  MapParticles3D(const MapParticles3D &st) = delete;
  /// Disable (implicit) copies.
  MapParticles3D &operator=(MapParticles3D const &a) = delete;

  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

  std::unique_ptr<MapParticlesCommon> map_particles_common;
  std::unique_ptr<MapParticlesHost> map_particles_host;
  std::unique_ptr<MapParticles3DRegular> map_particles_3d_regular;

  int count_regular = 0;
  int count_deformed = 0;

public:
  /**
   *  Constructor for mapping class.
   *
   *  @param sycl_target SYCLTarget on which to perform mapping.
   *  @param particle_mesh_interface ParticleMeshInterface containing 2D
   * Nektar++ cells.
   */
  MapParticles3D(SYCLTargetSharedPtr sycl_target,
                 ParticleMeshInterfaceSharedPtr particle_mesh_interface)
      : sycl_target(sycl_target),
        particle_mesh_interface(particle_mesh_interface) {

    this->map_particles_common =
        std::make_unique<MapParticlesCommon>(sycl_target);

    this->map_particles_host = std::make_unique<MapParticlesHost>(
        sycl_target, particle_mesh_interface);

    this->map_particles_3d_regular = std::make_unique<MapParticles3DRegular>(
        sycl_target, particle_mesh_interface);

    // determine if there are regular and deformed geometry objects
    this->count_regular = 0;
    this->count_deformed = 0;
    {
      std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
          geoms_local_tmp;
      get_all_elements_3d(particle_mesh_interface->graph, geoms_local_tmp);
      count_geometry_types(geoms_local_tmp, &count_regular, &count_deformed);
      count_geometry_types(particle_mesh_interface->remote_geoms_3d,
                           &count_regular, &count_deformed);
    }
  }

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  triangles and quads.
   */
  inline void map(ParticleGroup &particle_group, const int map_cell = -1,
                  const double tol = 0.0) {

    if (this->count_regular > 0) {
      // attempt to bin particles into regular geometry objects
      this->map_particles_3d_regular->map(particle_group, map_cell, tol);
    }

    bool particles_not_mapped = true;
    if (this->count_deformed > 0) {

      // are there particles whcih are not yet mapped into cells
      particles_not_mapped = this->map_particles_common->check_map(
          particle_group, map_cell, false);

      // attempt to bin the remaining particles into deformed cells if there are
      // deformed cells.
      if (particles_not_mapped) {
        this->map_particles_host->map(particle_group, map_cell, tol);
      }
    }

    // if there are particles not yet mapped this may be an error depending on
    // which stage of NESO-Particles hybrid move we are at.
    particles_not_mapped =
        this->map_particles_common->check_map(particle_group, map_cell, true);

    NESOASSERT(!particles_not_mapped,
               "Failed to find cell containing one or more particles.");
  }
};

} // namespace NESO

#endif
