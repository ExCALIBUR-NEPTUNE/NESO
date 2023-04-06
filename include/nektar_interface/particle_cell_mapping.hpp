#ifndef __PARTICLE_CELL_MAPPING_H__
#define __PARTICLE_CELL_MAPPING_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "candidate_cell_mapping.hpp"
#include "particle_mesh_interface.hpp"

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

#ifndef MAPPING_CROSS_PRODUCT_3D
#define MAPPING_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)           \
  c1 = ((a2) * (b3)) - ((a3) * (b2));                                          \
  c2 = ((a3) * (b1)) - ((a1) * (b3));                                          \
  c3 = ((a1) * (b2)) - ((a2) * (b1));
#endif

#ifndef MAPPING_DOT_PRODUCT_3D
#define MAPPING_DOT_PRODUCT_3D(a1, a2, a3, b1, b2, b3)                         \
  ((a1) * (b1) + (a2) * (b2) + (a3) * (b3))
#endif

/**
 *  Map a global coordinate to a local coordinate in reference space (xi).
 *
 *  @param geom 2D Geometry object to map.
 *  @param coords Global coordinates of point (physical space).
 *  @param Lcoords Local coordinates (xi) in reference space.
 *  @returns Maximum distance from geometry object to point (in refence space)
 * if not contained.
 */
template <typename T>
inline double get_local_coords_2d(std::shared_ptr<T> geom,
                                  const Array<OneD, const NekDouble> &coords,
                                  Array<OneD, NekDouble> &Lcoords) {

  NESOASSERT(geom->GetMetricInfo()->GetGtype() == eRegular,
             "Not a regular geometry object");

  int last_point_index;
  if (geom->GetShapeType() == LibUtilities::eTriangle) {
    last_point_index = 2;
  } else if (geom->GetShapeType() == LibUtilities::eQuadrilateral) {
    last_point_index = 3;
  } else {
    NESOASSERT(false, "get_local_coords_2d Unknown shape type.");
  }

  NESOASSERT(geom->GetCoordim() == 2, "Expected coordim == 2");

  const double last_coord = (geom->GetCoordim() == 3) ? coords[2] : 0.0;
  const double r0 = coords[0];
  const double r1 = coords[1];
  const double r2 = last_coord;

  const auto v0 = geom->GetVertex(0);
  const auto v1 = geom->GetVertex(1);
  const auto v2 = geom->GetVertex(last_point_index);

  const double er_0 = r0 - (*v0)[0];
  const double er_1 = r1 - (*v0)[1];
  const double er_2 = (v0->GetCoordim() == 3) ? r2 - (*v0)[2] : 0.0;

  const double e10_0 = (*v1)[0] - (*v0)[0];
  const double e10_1 = (*v1)[1] - (*v0)[1];
  const double e10_2 = (v0->GetCoordim() == 3 && v1->GetCoordim() == 3)
                           ? (*v1)[2] - (*v0)[2]
                           : 0.0;

  const double e20_0 = (*v2)[0] - (*v0)[0];
  const double e20_1 = (*v2)[1] - (*v0)[1];
  const double e20_2 = (v0->GetCoordim() == 3 && v2->GetCoordim() == 3)
                           ? (*v2)[2] - (*v0)[2]
                           : 0.0;

  MAPPING_CROSS_PRODUCT_3D(e10_0, e10_1, e10_2, e20_0, e20_1, e20_2,
                           const double norm_0, const double norm_1,
                           const double norm_2)
  MAPPING_CROSS_PRODUCT_3D(norm_0, norm_1, norm_2, e10_0, e10_1, e10_2,
                           const double orth1_0, const double orth1_1,
                           const double orth1_2)
  MAPPING_CROSS_PRODUCT_3D(norm_0, norm_1, norm_2, e20_0, e20_1, e20_2,
                           const double orth2_0, const double orth2_1,
                           const double orth2_2)

  const double scale0 =
      MAPPING_DOT_PRODUCT_3D(er_0, er_1, er_2, orth2_0, orth2_1, orth2_2) /
      MAPPING_DOT_PRODUCT_3D(e10_0, e10_1, e10_2, orth2_0, orth2_1, orth2_2);
  Lcoords[0] = 2.0 * scale0 - 1.0;
  const double scale1 =
      MAPPING_DOT_PRODUCT_3D(er_0, er_1, er_2, orth1_0, orth1_1, orth1_2) /
      MAPPING_DOT_PRODUCT_3D(e20_0, e20_1, e20_2, orth1_0, orth1_1, orth1_2);
  Lcoords[1] = 2.0 * scale1 - 1.0;

  double eta0, eta1;
  if (geom->GetShapeType() == LibUtilities::eTriangle) {
    NekDouble d1 = 1. - Lcoords[1];
    if (fabs(d1) < NekConstants::kNekZeroTol) {
      if (d1 >= 0.) {
        d1 = NekConstants::kNekZeroTol;
      } else {
        d1 = -NekConstants::kNekZeroTol;
      }
    }
    eta0 = 2. * (1. + Lcoords[0]) / d1 - 1.0;
    eta1 = Lcoords[1];

  } else if (geom->GetShapeType() == LibUtilities::eQuadrilateral) {
    eta0 = Lcoords[0];
    eta1 = Lcoords[1];
  }

  double dist = 0.0;
  bool contained =
      ((eta0 <= 1.0) && (eta0 >= -1.0) && (eta1 <= 1.0) && (eta1 >= -1.0));
  if (!contained) {
    dist = (eta0 < -1.0) ? (-1.0 - eta0) : 0.0;
    dist = std::max(dist, (eta0 > 1.0) ? (eta0 - 1.0) : 0.0);
    dist = std::max(dist, (eta1 < -1.0) ? (-1.0 - eta1) : 0.0);
    dist = std::max(dist, (eta1 > 1.0) ? (eta1 - 1.0) : 0.0);
  }

  return dist;
}

/**
 *  Test if a 2D Geometry object contains a point. Returns the computed
 * reference coordinate (xi).
 *
 *  @param geom 2D Geometry object, e.g. QuadGeom, TriGeom.
 *  @param global_coord Global coordinate to map to local coordinate.
 *  @param local_coord Output, computed locate coordinate in reference space.
 *  @param tol Input tolerance for geometry containing point.
 */
template <typename T>
inline bool
contains_point_2d(std::shared_ptr<T> geom, Array<OneD, NekDouble> &global_coord,
                  Array<OneD, NekDouble> &local_coord, const NekDouble tol) {

  const double dist = get_local_coords_2d(geom, global_coord, local_coord);
  bool contained = dist <= tol;
  return contained;
}

/**
 * Class to map particle positions to Nektar++ cells. Implemented for triangles
 * and quads.
 */
class NektarGraphLocalMapperT : public LocalMapper {
private:
  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  const double tol;

  std::unique_ptr<CandidateCellMapper> candidate_cell_mapper;
  std::unique_ptr<ErrorPropagate> ep;

public:
  ~NektarGraphLocalMapperT(){};

  /**
   * Callback for ParticleGroup to execute for additional setup of the
   * LocalMapper that may involve the ParticleGroup.
   *
   * @param particle_group ParticleGroup instance.
   */
  inline void particle_group_callback(ParticleGroup &particle_group) {

    particle_group.add_particle_dat(
        ParticleDat(particle_group.sycl_target,
                    ParticleProp(Sym<REAL>("NESO_REFERENCE_POSITIONS"),
                                 particle_group.domain->mesh->get_ndim()),
                    particle_group.domain->mesh->get_cell_count()));
  };

  /**
   *  Construct a new mapper object.
   *
   *  @param sycl_target SYCLTarget to use.
   *  @param particle_mesh_interface Interface between NESO-Particles and
   * Nektar++ mesh.
   *  @param tol Tolerance to pass to Nektar++ to bin particles into cells.
   */
  NektarGraphLocalMapperT(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      const double tol = 1.0e-10)
      : sycl_target(sycl_target),
        particle_mesh_interface(particle_mesh_interface), tol(tol) {
    this->candidate_cell_mapper = std::make_unique<CandidateCellMapper>(
        this->sycl_target, this->particle_mesh_interface);
    this->ep = std::make_unique<ErrorPropagate>(this->sycl_target);
  };

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  triangles and quads.
   */
  inline void map(ParticleGroup &particle_group, const int map_cell = -1) {

    auto &ccm = this->candidate_cell_mapper;

    // Get kernel pointers to the mesh data.
    const auto &mesh = ccm->cartesian_mesh;
    const auto k_mesh_origin0 = mesh->dh_origin->h_buffer.ptr[0];
    const auto k_mesh_origin1 = mesh->dh_origin->h_buffer.ptr[1];
    const auto k_mesh_cell_counts0 = mesh->dh_cell_counts->h_buffer.ptr[0];
    const auto k_mesh_cell_counts1 = mesh->dh_cell_counts->h_buffer.ptr[1];
    const auto k_mesh_inverse_cell_widths0 =
        mesh->dh_inverse_cell_widths->h_buffer.ptr[0];
    const auto k_mesh_inverse_cell_widths1 =
        mesh->dh_inverse_cell_widths->h_buffer.ptr[1];

    // Get kernel pointers to the map data.
    const auto k_map_cell_ids = ccm->dh_cell_ids->d_buffer.ptr;
    const auto k_map_mpi_ranks = ccm->dh_mpi_ranks->d_buffer.ptr;
    const auto k_map_type = ccm->dh_type->d_buffer.ptr;
    const auto k_map_vertices = ccm->dh_vertices->d_buffer.ptr;
    const auto k_map = ccm->dh_map->d_buffer.ptr;
    const auto k_map_sizes = ccm->dh_map_sizes->d_buffer.ptr;
    const auto k_map_stride = ccm->map_stride;
    const int k_geom_is_triangle = ccm->index_tri_geom;
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
    const size_t local_size = 256;
    const auto div_mod = std::div(max_cell_occupancy, local_size);
    const int outer_size = div_mod.quot + (div_mod.rem == 0 ? 0 : 1);
    const size_t cell_count =
        (map_cell > -1) ? 1
                        : static_cast<size_t>(position_dat->cell_dat.ncells);
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

                    const double r0 = p0;
                    const double r1 = p1;

                    bool cell_found = false;
                    for (int candidate_cell = 0;
                         candidate_cell < k_map_sizes[linear_mesh_cell];
                         candidate_cell++) {
                      const int geom_map_index =
                          k_map[linear_mesh_cell * k_map_stride +
                                candidate_cell];

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

                      MAPPING_CROSS_PRODUCT_3D(
                          e10_0, e10_1, e10_2, e20_0, e20_1, e20_2,
                          const double norm_0, const double norm_1,
                          const double norm_2)
                      MAPPING_CROSS_PRODUCT_3D(
                          norm_0, norm_1, norm_2, e10_0, e10_1, e10_2,
                          const double orth1_0, const double orth1_1,
                          const double orth1_2)
                      MAPPING_CROSS_PRODUCT_3D(
                          norm_0, norm_1, norm_2, e20_0, e20_1, e20_2,
                          const double orth2_0, const double orth2_1,
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
                        dist =
                            std::max(dist, (eta0 > 1.0) ? (eta0 - 1.0) : 0.0);
                        dist =
                            std::max(dist, (eta1 < -1.0) ? (-1.0 - eta1) : 0.0);
                        dist =
                            std::max(dist, (eta1 > 1.0) ? (eta1 - 1.0) : 0.0);
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

                    // if a geom is not found and there is a non-null global MPI
                    // rank then this function was called after the global move
                    // and the lack of a local cell / mpi rank is a fatal error.
                    if (((k_part_mpi_ranks)[cellx][0][layerx] > -1) &&
                        !cell_found) {
                      NESO_KERNEL_ASSERT(false, k_ep);
                    }
                  }
                }
              });
        })
        .wait_and_throw();

    NESOASSERT(!this->ep->get_flag(),
               "Failed to bin particle into local cell.");
  }

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  triangles and quads.
   */
  inline void map_host(ParticleGroup &particle_group, const int map_cell = -1) {

    ParticleDatSharedPtr<REAL> &position_dat = particle_group.position_dat;
    ParticleDatSharedPtr<REAL> &ref_position_dat =
        particle_group[Sym<REAL>("NESO_REFERENCE_POSITIONS")];
    ParticleDatSharedPtr<INT> &cell_id_dat = particle_group.cell_id_dat;
    ParticleDatSharedPtr<INT> &mpi_rank_dat = particle_group.mpi_rank_dat;

    auto t0 = profile_timestamp();
    const int rank = this->sycl_target->comm_pair.rank_parent;
    const int ndim = this->particle_mesh_interface->ndim;
    const int ncell = this->particle_mesh_interface->get_cell_count();
    const int nrow_max = mpi_rank_dat->cell_dat.get_nrow_max();
    auto graph = this->particle_mesh_interface->graph;

    CellDataT<REAL> particle_positions(sycl_target, nrow_max,
                                       position_dat->ncomp);
    CellDataT<REAL> ref_particle_positions(sycl_target, nrow_max,
                                           ref_position_dat->ncomp);
    CellDataT<INT> mpi_ranks(sycl_target, nrow_max, mpi_rank_dat->ncomp);
    CellDataT<INT> cell_ids(sycl_target, nrow_max, cell_id_dat->ncomp);

    EventStack event_stack;

    int cell_start, cell_end;
    if (map_cell < 0) {
      cell_start = 0;
      cell_end = ncell;
    } else {
      cell_start = map_cell;
      cell_end = map_cell + 1;
    }

    double time_copy_from = 0.0;
    double time_copy_to = 0.0;
    double time_map_nektar = 0.0;
    double time_halo_lookup = 0.0;

    for (int cellx = cell_start; cellx < cell_end; cellx++) {

      auto t0_copy_from = profile_timestamp();
      position_dat->cell_dat.get_cell_async(cellx, particle_positions,
                                            event_stack);
      ref_position_dat->cell_dat.get_cell_async(cellx, ref_particle_positions,
                                                event_stack);
      mpi_rank_dat->cell_dat.get_cell_async(cellx, mpi_ranks, event_stack);
      cell_id_dat->cell_dat.get_cell_async(cellx, cell_ids, event_stack);

      event_stack.wait();
      time_copy_from += profile_elapsed(t0_copy_from, profile_timestamp());

      const int nrow = mpi_rank_dat->cell_dat.nrow[cellx];
      Array<OneD, NekDouble> global_coord(3);
      Array<OneD, NekDouble> local_coord(3);
      auto point = std::make_shared<PointGeom>(ndim, -1, 0.0, 0.0, 0.0);

      for (int rowx = 0; rowx < nrow; rowx++) {

        if ((mpi_ranks)[1][rowx] < 0) {

          // copy the particle position into a nektar++ point format
          for (int dimx = 0; dimx < ndim; dimx++) {
            global_coord[dimx] = particle_positions[dimx][rowx];
            local_coord[dimx] = ref_particle_positions[dimx][rowx];
          }

          // update the PointGeom
          point->UpdatePosition(global_coord[0], global_coord[1],
                                global_coord[2]);

          auto t0_nektar_lookup = profile_timestamp();
          // get the elements that could contain the point
          auto element_ids = graph->GetElementsContainingPoint(point);
          // test the possible local geometry elements

          bool geom_found = false;
          // check the original nektar++ geoms
          for (auto &ex : element_ids) {
            Geometry2DSharedPtr geom_2d = graph->GetGeometry2D(ex);

            geom_found = contains_point_2d(geom_2d, global_coord, local_coord,
                                           this->tol);
            if (geom_found) {
              (mpi_ranks)[1][rowx] = rank;
              (cell_ids)[0][rowx] = ex;
              for (int dimx = 0; dimx < ndim; dimx++) {
                ref_particle_positions[dimx][rowx] = local_coord[dimx];
              }
              break;
            }
          }
          time_map_nektar +=
              profile_elapsed(t0_nektar_lookup, profile_timestamp());

          auto t0_halo_lookup = profile_timestamp();
          // containing geom not found in the set of owned geoms, now check the
          // remote geoms
          if (!geom_found) {
            for (auto &remote_geom :
                 this->particle_mesh_interface->remote_triangles) {

              geom_found = contains_point_2d(remote_geom->geom, global_coord,
                                             local_coord, this->tol);
              if (geom_found) {
                (mpi_ranks)[1][rowx] = remote_geom->rank;
                (cell_ids)[0][rowx] = remote_geom->id;
                for (int dimx = 0; dimx < ndim; dimx++) {
                  ref_particle_positions[dimx][rowx] = local_coord[dimx];
                }
                break;
              }
            }
          }
          if (!geom_found) {
            for (auto &remote_geom :
                 this->particle_mesh_interface->remote_quads) {

              geom_found = contains_point_2d(remote_geom->geom, global_coord,
                                             local_coord, this->tol);
              if (geom_found) {
                (mpi_ranks)[1][rowx] = remote_geom->rank;
                (cell_ids)[0][rowx] = remote_geom->id;
                for (int dimx = 0; dimx < ndim; dimx++) {
                  ref_particle_positions[dimx][rowx] = local_coord[dimx];
                }
                break;
              }
            }
          }
          time_halo_lookup +=
              profile_elapsed(t0_halo_lookup, profile_timestamp());

          // if a geom is not found and there is a non-null global MPI rank then
          // this function was called after the global move and the lack of a
          // local cell / mpi rank is a fatal error.
          if (((mpi_ranks)[0][rowx] > -1) && !geom_found) {
            NESOASSERT(false, "No local geometry found for particle");
          }
        }
      }

      auto t0_copy_to = profile_timestamp();
      ref_position_dat->cell_dat.set_cell_async(cellx, ref_particle_positions,
                                                event_stack);
      mpi_rank_dat->cell_dat.set_cell_async(cellx, mpi_ranks, event_stack);
      cell_id_dat->cell_dat.set_cell_async(cellx, cell_ids, event_stack);
      event_stack.wait();

      time_copy_to = profile_elapsed(t0_copy_to, profile_timestamp());
    }

    sycl_target->profile_map.inc("NektarGraphLocalMapperT", "copy_to", 0,
                                 time_copy_to);
    sycl_target->profile_map.inc("NektarGraphLocalMapperT", "map_halo", 0,
                                 time_halo_lookup);
    sycl_target->profile_map.inc("NektarGraphLocalMapperT", "map_nektar", 0,
                                 time_map_nektar);
    sycl_target->profile_map.inc("NektarGraphLocalMapperT", "copy_from", 0,
                                 time_copy_from);
    sycl_target->profile_map.inc("NektarGraphLocalMapperT", "map", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };
};

} // namespace NESO

#endif
