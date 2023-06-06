#ifndef __PARTICLE_CELL_MAPPING_NEWTON_H__
#define __PARTICLE_CELL_MAPPING_NEWTON_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "candidate_cell_mapping.hpp"
#include "particle_cell_mapping_common.hpp"

#include <LibUtilities/BasicConst/NektarUnivConsts.hpp>
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {
namespace Newton {

/**
 *  Abstract base class for Newton iteration methods for binning particles into
 *  Nektar++ cells. Subclasses must be device copyable.
 */
template <typename SPECIALISATION> struct MappingNewtonIterationBase {

  /**
   *  Specialisations should write to the pointer a struct that contains all the
   *  data which will be required to perform a Newton iteration on the SYCL
   *  device.
   *
   *  @param geom A geometry object which particles may be binned into.
   *  @param data_host A host pointer to a buffer which will be kept on the
   *  host.
   *  @param data_device A host pointer to a buffer which will be copied to the
   *  compute device.
   *
   */
  inline void write_data(GeometrySharedPtr geom, void *data_host,
                         void *data_device) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.write_data_v(geom, data_host, data_device);
  }

  /**
   *  Called at destruction to enable implementations to free additional memory
   *  which was allocated for variable length Newton iteration data (and
   *  pointed to in the data_device memory).
   */
  inline void free_data(void *data_host) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.free_data_v(data_host);
  }

  /**
   *  The number of bytes required to store the data required to perform a
   *  Newton iteration on the device. i.e. the write_data call is free to write
   *  this number of bytes to the passed pointer on the host (and copied to
   *  device).
   *
   *  @returns Number of bytes required to be allocated.
   */
  inline size_t data_size_host() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.data_size_host_v();
  }

  /**
   *  The number of bytes required to store the data required to perform a
   *  Newton iteration on the host. i.e. the write_data call is free to write
   *  this number of bytes to the passed pointer on the host.
   *
   *  @returns Number of bytes required to be allocated.
   */
  inline size_t data_size_device() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.data_size_device_v();
  }

  /**
   * Perform a Newton iteration such that
   *
   * xi_{n+1} = xi_n - J^{-1}(xi_n) F(xi_n)
   *
   * where
   *
   * F(xi) = X(xi) - P
   *
   * for target physical coordinates P. All data required to perform the
   * iteration should be contained in the memory region (i.e. a struct) pointed
   * to by d_data.
   *
   * @param[in] d_data Pointer to data required to perform the Newton iteration.
   * @param[in] xi0 Current iteration of xi coordinate, x component.
   * @param[in] xi1 Current iteration of xi coordinate, y component.
   * @param[in] xi2 Current iteration of xi coordinate, z component.
   * @param[in] phys0 Target coordinate in physical space, x component.
   * @param[in] phys1 Target coordinate in physical space, y component.
   * @param[in] phys2 Target coordinate in physical space, z component.
   * @param[in] f0 F(xi), x component.
   * @param[in] f1 F(xi), y component.
   * @param[in] f2 F(xi), z component.
   * @param[in, out] xin0 Output new iteration for local coordinate xi, x
   * component.
   * @param[in, out] xin1 Output new iteration for local coordinate xi, y
   * component.
   * @param[in, out] xin2 Output new iteration for local coordinate xi, z
   * component.
   */
  inline void newton_step(const void *d_data, const REAL xi0, const REAL xi1,
                          const REAL xi2, const REAL phys0, const REAL phys1,
                          const REAL phys2, const REAL f0, const REAL f1,
                          const REAL f2, REAL *xin0, REAL *xin1, REAL *xin2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.newton_step_v(d_data, xi0, xi1, xi2, phys0, phys1, phys2, f0, f1,
                             f2, xin0, xin1, xin2);
  }

  /**
   * Compute residual
   *
   * ||F(xi_n)||
   *
   * where
   *
   * F(xi) = X(xi) - P
   *
   * for target physical coordinates P and sensible norm ||F||. All data
   * required to compute the residual should be contained in the memory region
   * (i.e. a struct) pointed to by d_data. Computes and returns F(xi) for the
   * passed xi.
   *
   * @param[in] d_data Pointer to data required to perform the Newton iteration.
   * @param[in] xi0 Current iteration of xi coordinate, x component.
   * @param[in] xi1 Current iteration of xi coordinate, y component.
   * @param[in] xi2 Current iteration of xi coordinate, z component.
   * @param[in] phys0 Target coordinate in physical space, x component.
   * @param[in] phys1 Target coordinate in physical space, y component.
   * @param[in] phys2 Target coordinate in physical space, z component.
   * @param[in, out] f0 F(xi), x component.
   * @param[in, out] f1 F(xi), y component.
   * @param[in, out] f2 F(xi), z component.
   * @returns Residual.
   */
  inline REAL newton_residual(const void *d_data, const REAL xi0,
                              const REAL xi1, const REAL xi2, const REAL phys0,
                              const REAL phys1, const REAL phys2, REAL *f0,
                              REAL *f1, REAL *f2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.newton_residual_v(d_data, xi0, xi1, xi2, phys0, phys1,
                                        phys2, f0, f1, f2);
  }

  /**
   * Get the number of coordinate dimentions the iteration is performed in.
   * i.e. how many position components should be read from the particle and
   * reference coordinates written.
   *
   *  @returns Number of coordinate dimensions.
   */
  inline int get_ndim() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.get_ndim_v();
  }

  /**
   * Set the initial iteration prior to the newton iteration.
   *
   * @param[in] d_data Pointer to data required to perform the Newton iteration.
   * @param[in] phys0 Target coordinate in physical space, x component.
   * @param[in] phys1 Target coordinate in physical space, y component.
   * @param[in] phys2 Target coordinate in physical space, z component.
   * @param[in,out] xi0 Input x coordinate to reset.
   * @param[in,out] xi1 Input y coordinate to reset.
   * @param[in,out] xi2 Input z coordinate to reset.
   */
  inline void set_initial_iteration(const void *d_data, const REAL phys0,
                                    const REAL phys1, const REAL phys2,
                                    REAL *xi0, REAL *xi1, REAL *xi2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.set_initial_iteration_v(d_data, phys0, phys1, phys2, xi0, xi1,
                                       xi2);
  }

  /**
   *  Map local coordinate (xi) to local collaposed coordinate (eta).
   *
   *
   * @param[in] d_data Pointer to data required to perform the Newton iteration.
   * @param[in] xi0 Local coordinate (xi) to be mapped to collapsed coordinate,
   * x component.
   * @param[in] xi1 Local coordinate (xi) to be mapped to collapsed coordinate,
   * y component.
   * @param[in] xi2 Local coordinate (xi) to be mapped to collapsed coordinate,
   * z component.
   * @param[in, out] eta0 Local collaposed coordinate (eta), x component.
   * @param[in, out] eta1 Local collaposed coordinate (eta), y component.
   * @param[in, out] eta2 Local collaposed coordinate (eta), z component.
   */
  inline void loc_coord_to_loc_collapsed(const void *d_data, const REAL xi0,
                                         const REAL xi1, const REAL xi2,
                                         REAL *eta0, REAL *eta1, REAL *eta2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loc_coord_to_loc_collapsed_v(d_data, xi0, xi1, xi2, eta0, eta1,
                                            eta2);
  }
};

/**
 *  Utility class to evaluate X maps and their inverse.
 */
template <typename NEWTON_TYPE> class XMapNewton {
protected:
  /// Disable (implicit) copies.
  XMapNewton(const XMapNewton &st) = delete;
  /// Disable (implicit) copies.
  XMapNewton &operator=(XMapNewton const &a) = delete;

  MappingNewtonIterationBase<NEWTON_TYPE> newton_type;

  SYCLTargetSharedPtr sycl_target;
  const size_t num_bytes_per_map_device;
  const size_t num_bytes_per_map_host;

  /// The data required to perform newton iterations for each geom on the
  /// device.
  std::unique_ptr<BufferDeviceHost<char>> dh_data;

  std::unique_ptr<BufferDeviceHost<REAL>> dh_fdata;
  /// The data required to perform newton iterations for each geom on the host.
  std::vector<char> h_data;

  template <typename U> inline void write_data(U &geom) {
    if (this->num_bytes_per_map_host) {
      this->h_data = std::vector<char>(this->num_bytes_per_map_host);
    }
    if (this->num_bytes_per_map_device) {
      this->dh_data = std::make_unique<BufferDeviceHost<char>>(
          this->sycl_target, this->num_bytes_per_map_device);
    }
    auto d_data_ptr = (this->num_bytes_per_map_device)
                          ? this->dh_data->h_buffer.ptr
                          : nullptr;
    auto h_data_ptr =
        (this->num_bytes_per_map_host) ? this->h_data.data() : nullptr;

    this->newton_type.write_data(geom, h_data_ptr, d_data_ptr);
    this->dh_data->host_to_device();
  }

public:
  ~XMapNewton() {
    auto h_data_ptr =
        (this->num_bytes_per_map_host) ? this->h_data.data() : nullptr;
    this->newton_type.free_data(h_data_ptr);
  }

  /**
   *  Create new instance from Newton implementation.
   *
   *  @param sycl_target SYCLTarget to use for computation.
   *  @param geom Nektar++ geometry type that matches the Newton method
   *  implementation.
   */
  template <typename TYPE_GEOM>
  XMapNewton(SYCLTargetSharedPtr sycl_target, std::shared_ptr<TYPE_GEOM> geom)
      : newton_type(MappingNewtonIterationBase<NEWTON_TYPE>()), sycl_target(sycl_target), num_bytes_per_map_host(newton_type.data_size_host()) ,num_bytes_per_map_device(newton_type.data_size_device()){
    this->write_data(geom);
    this->dh_fdata =
        std::make_unique<BufferDeviceHost<REAL>>(this->sycl_target, 3);
  }

  /**
   * For a reference position xi compute the global position X(xi).
   *
   * @param[in] xi0 Reference position, x component.
   * @param[in] xi1 Reference position, y component.
   * @param[in] xi2 Reference position, z component.
   * @param[in, out] xi0 Global position X(xi), x component.
   * @param[in, out] xi1 Global position X(xi), y component.
   * @param[in, out] xi2 Global position X(xi), z component.
   */
  inline void x(const REAL xi0, const REAL xi1, const REAL xi2,
                      REAL *phys0, REAL *phys1, REAL *phys2) {

    auto k_map_data = this->dh_data->d_buffer.ptr;
    auto k_fdata = this->dh_fdata->d_buffer.ptr;

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task<>([=]() {
            MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};

            REAL f0 = 0.0;
            REAL f1 = 0.0;
            REAL f2 = 0.0;
            const REAL p0 = 0.0;
            const REAL p1 = 0.0;
            const REAL p2 = 0.0;

            k_newton_type.newton_residual(k_map_data, xi0, xi1, xi2, p0, p1, p2,
                                          &f0, &f1, &f2);

            k_fdata[0] = f0;
            k_fdata[1] = f1;
            k_fdata[2] = f2;
          });
        })
        .wait_and_throw();

    this->dh_fdata->device_to_host();
    *phys0 = this->dh_fdata->h_buffer.ptr[0];
    *phys1 = this->dh_fdata->h_buffer.ptr[1];
    *phys2 = this->dh_fdata->h_buffer.ptr[2];
  }


};





/**
 *  TODO
 */
template <typename NEWTON_TYPE> class MapParticlesNewton {
protected:
  /// Disable (implicit) copies.
  MapParticlesNewton(const MapParticlesNewton &st) = delete;
  /// Disable (implicit) copies.
  MapParticlesNewton &operator=(MapParticlesNewton const &a) = delete;

  SYCLTargetSharedPtr sycl_target;
  std::unique_ptr<CoarseLookupMap> coarse_lookup_map;

  /// Number of geometry objects this instance may map to.
  int num_geoms;
  /// Number of coordinate dimensions.
  const int ndim;
  /// The nektar++ cell id for the cells indices pointed to from the map.
  std::unique_ptr<BufferDeviceHost<int>> dh_cell_ids;
  /// The MPI rank that owns the cell.
  std::unique_ptr<BufferDeviceHost<int>> dh_mpi_ranks;
  /// The type of the cell, e.g. a quad or a triangle.
  std::unique_ptr<BufferDeviceHost<int>> dh_type;
  /// The data required to perform newton iterations for each geom on the
  /// device.
  std::unique_ptr<BufferDeviceHost<char>> dh_data;
  /// The data required to perform newton iterations for each geom on the host.
  std::vector<char> h_data;
  /// Instance to throw kernel errors with.
  std::unique_ptr<ErrorPropagate> ep;
  /// The Newton iteration class.
  MappingNewtonIterationBase<NEWTON_TYPE> newton_type;
  const size_t num_bytes_per_map_device;
  const size_t num_bytes_per_map_host;

  template <typename U> inline void write_data(U &geom, const int index) {

    auto d_data_ptr = (this->num_bytes_per_map_device)
                          ? this->dh_data->h_buffer.ptr +
                                index * this->num_bytes_per_map_device
                          : nullptr;
    auto h_data_ptr =
        (this->num_bytes_per_map_host)
            ? this->h_data.data() + index * this->num_bytes_per_map_host
            : nullptr;

    this->newton_type.write_data(geom, h_data_ptr, d_data_ptr);
  }

public:
  ~MapParticlesNewton() {
    for (int index = 0; index < num_geoms; index++) {
      auto h_data_ptr =
          (this->num_bytes_per_map_host)
              ? this->h_data.data() + index * this->num_bytes_per_map_host
              : nullptr;
      this->newton_type.free_data(h_data_ptr);
    }
  }

  /**
   *  TODO
   */
  template <typename TYPE_LOCAL, typename TYPE_REMOTE>
  MapParticlesNewton(MappingNewtonIterationBase<NEWTON_TYPE> newton_type,
                     SYCLTargetSharedPtr sycl_target,
                     std::map<int, std::shared_ptr<TYPE_LOCAL>> &geoms_local,
                     std::vector<std::shared_ptr<TYPE_REMOTE>> &geoms_remote)
      : newton_type(newton_type), sycl_target(sycl_target),
        num_bytes_per_map_device(newton_type.data_size_device()),
        num_bytes_per_map_host(newton_type.data_size_host()),
        ndim(newton_type.get_ndim()) {

    this->num_geoms = geoms_local.size() + geoms_remote.size();
    if (this->num_geoms > 0) {

      // create the coarse lookup mesh
      this->coarse_lookup_map = std::make_unique<CoarseLookupMap>(
          this->ndim, this->sycl_target, geoms_local, geoms_remote);

      // store the information required to evaluate v_GetLocCoords for regular
      // Geometry3D objects.
      // map from cartesian cells to nektar mesh cells
      std::map<int, std::list<std::pair<double, int>>> geom_map;
      this->dh_cell_ids =
          std::make_unique<BufferDeviceHost<int>>(this->sycl_target, num_geoms);
      this->dh_mpi_ranks =
          std::make_unique<BufferDeviceHost<int>>(this->sycl_target, num_geoms);
      this->dh_type =
          std::make_unique<BufferDeviceHost<int>>(this->sycl_target, num_geoms);

      if (this->num_bytes_per_map_device) {
        this->dh_data = std::make_unique<BufferDeviceHost<char>>(
            this->sycl_target, num_geoms * this->num_bytes_per_map_device);
      }
      if (this->num_bytes_per_map_host) {
        this->h_data =
            std::vector<char>(num_geoms * this->num_bytes_per_map_host);
      }

      const int index_tri = shape_type_to_int(eTriangle);
      const int index_quad = shape_type_to_int(eQuadrilateral);
      const int index_tet = shape_type_to_int(eTetrahedron);
      const int index_pyr = shape_type_to_int(ePyramid);
      const int index_prism = shape_type_to_int(ePrism);
      const int index_hex = shape_type_to_int(eHexahedron);

      const int rank = this->sycl_target->comm_pair.rank_parent;

      for (auto &geom : geoms_local) {
        const int id = geom.second->GetGlobalID();
        const int cell_index = this->coarse_lookup_map->gid_to_lookup_id.at(id);
        NESOASSERT((cell_index < num_geoms) && (0 <= cell_index),
                   "Bad cell index from map.");
        NESOASSERT(id == geom.first, "ID mismatch");

        this->dh_cell_ids->h_buffer.ptr[cell_index] = id;
        this->dh_mpi_ranks->h_buffer.ptr[cell_index] = rank;
        const int geom_type = shape_type_to_int(geom.second->GetShapeType());
        NESOASSERT((geom_type == index_tet) || (geom_type == index_pyr) ||
                       (geom_type == index_prism) || (geom_type == index_hex) ||
                       (geom_type == index_tri) || (geom_type == index_quad),
                   "Unknown shape type.");
        this->dh_type->h_buffer.ptr[cell_index] = geom_type;
        this->write_data(geom.second, cell_index);
      }

      for (auto &geom : geoms_remote) {
        const int id = geom->id;
        const int cell_index = this->coarse_lookup_map->gid_to_lookup_id.at(id);
        NESOASSERT((cell_index < num_geoms) && (0 <= cell_index),
                   "Bad cell index from map.");
        this->dh_cell_ids->h_buffer.ptr[cell_index] = id;
        this->dh_mpi_ranks->h_buffer.ptr[cell_index] = geom->rank;
        const int geom_type = shape_type_to_int(geom->geom->GetShapeType());
        NESOASSERT((geom_type == index_tet) || (geom_type == index_pyr) ||
                       (geom_type == index_prism) || (geom_type == index_hex) ||
                       (geom_type == index_tri) || (geom_type == index_quad),
                   "Unknown shape type.");
        this->dh_type->h_buffer.ptr[cell_index] = geom_type;
        this->write_data(geom->geom, cell_index);
      }

      this->dh_cell_ids->host_to_device();
      this->dh_mpi_ranks->host_to_device();
      this->dh_type->host_to_device();
      this->dh_data->host_to_device();
    }
    this->ep = std::make_unique<ErrorPropagate>(this->sycl_target);
  }

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  Geometry objects via Newton iteration.
   */
  inline void map(ParticleGroup &particle_group, const int map_cell = -1,
                  const double tol = 1.0e-10) {

    if (this->num_geoms == 0) {
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
    const auto k_map_data = this->dh_data->d_buffer.ptr;
    const auto k_map = clm->dh_map->d_buffer.ptr;
    const auto k_map_sizes = clm->dh_map_sizes->d_buffer.ptr;
    const auto k_map_stride = clm->map_stride;
    const double k_tol = tol;
    const int k_ndim = this->ndim;
    const int k_num_bytes_per_map_device = this->num_bytes_per_map_device;
    const int k_max_iterations = 51;

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
                    const REAL p0 = k_part_positions[cellx][0][layerx];
                    const REAL p1 =
                        (k_ndim > 1) ? k_part_positions[cellx][1][layerx] : 0.0;
                    const REAL p2 =
                        (k_ndim > 2) ? k_part_positions[cellx][2][layerx] : 0.0;
                    const REAL shifted_p0 = p0 - k_mesh_origin[0];
                    const REAL shifted_p1 =
                        (k_ndim > 1) ? p1 - k_mesh_origin[1] : 0.0;
                    const REAL shifted_p2 =
                        (k_ndim > 2) ? p2 - k_mesh_origin[2] : 0.0;

                    // determine the cartesian mesh cell for the position
                    int c0 = (k_mesh_inverse_cell_widths[0] * shifted_p0);
                    int c1 = (k_ndim > 1)
                                 ? (k_mesh_inverse_cell_widths[1] * shifted_p1)
                                 : 0;
                    int c2 = (k_ndim > 2)
                                 ? (k_mesh_inverse_cell_widths[2] * shifted_p2)
                                 : 0;
                    c0 = (c0 < 0) ? 0 : c0;
                    c1 = (c1 < 0) ? 0 : c1;
                    c2 = (c2 < 0) ? 0 : c2;
                    c0 = (c0 >= k_mesh_cell_counts[0])
                             ? k_mesh_cell_counts[0] - 1
                             : c0;
                    if (k_ndim > 1) {
                      c1 = (c1 >= k_mesh_cell_counts[1])
                               ? k_mesh_cell_counts[1] - 1
                               : c1;
                    }
                    if (k_ndim > 2) {
                      c2 = (c2 >= k_mesh_cell_counts[2])
                               ? k_mesh_cell_counts[2] - 1
                               : c2;
                    }

                    const int mcc0 = k_mesh_cell_counts[0];
                    const int mcc1 = (k_ndim > 1) ? k_mesh_cell_counts[1] : 0;
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

                      const char *map_data =
                          (k_num_bytes_per_map_device)
                              ? &k_map_data[geom_map_index *
                                            k_num_bytes_per_map_device]
                              : nullptr;

                      REAL xi0;
                      REAL xi1;
                      REAL xi2;
                      MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};
                      k_newton_type.set_initial_iteration(map_data, p0, p1, p2,
                                                          &xi0, &xi1, &xi2);

                      // Start of Newton iteration
                      REAL xin0, xin1, xin2;
                      REAL f0, f1, f2;

                      REAL residual = k_newton_type.newton_residual(
                          map_data, xi0, xi1, xi2, p0, p1, p2, &f0, &f1, &f2);

                      bool diverged = false;

                      for (int stepx = 0; ((stepx < k_max_iterations) &&
                                           (residual > k_tol) && (!diverged));
                           stepx++) {
                        k_newton_type.newton_step(map_data, xi0, xi1, xi2, p0,
                                                  p1, p2, f0, f1, f2, &xin0,
                                                  &xin1, &xin2);

                        xi0 = xin0;
                        xi1 = xin1;
                        xi2 = xin2;

                        residual = k_newton_type.newton_residual(
                            map_data, xi0, xi1, xi2, p0, p1, p2, &f0, &f1, &f2);

                        diverged = (ABS(xi0) > 15.0) || (ABS(xi1) > 15.0) ||
                                   (ABS(xi2) > 15.0);
                      }

                      bool converged = (residual <= tol);
                      REAL eta0;
                      REAL eta1;
                      REAL eta2;

                      k_newton_type.loc_coord_to_loc_collapsed(
                          map_data, xi0, xi1, xi2, &eta0, &eta1, &eta2);

                      bool contained =
                          ((eta0 <= 1.0) && (eta0 >= -1.0) && (eta1 <= 1.0) &&
                           (eta1 >= -1.0) && (eta2 <= 1.0) && (eta2 >= -1.0) &&
                           converged);

                      REAL dist = 0.0;
                      if ((!contained) && converged) {
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

                      cell_found = (dist <= k_tol) && converged;
                      if (cell_found) {
                        const int geom_id = k_map_cell_ids[geom_map_index];
                        const int mpi_rank = k_map_mpi_ranks[geom_map_index];
                        k_part_cell_ids[cellx][0][layerx] = geom_id;
                        k_part_mpi_ranks[cellx][1][layerx] = mpi_rank;
                        k_part_ref_positions[cellx][0][layerx] = xi0;
                        if (k_ndim > 1) {
                          k_part_ref_positions[cellx][1][layerx] = xi1;
                        }
                        if (k_ndim > 2) {
                          k_part_ref_positions[cellx][2][layerx] = xi2;
                        }
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

} // namespace Newton
} // namespace NESO

#endif
