#ifndef __PARTICLE_CELL_MAPPING_H__
#define __PARTICLE_CELL_MAPPING_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "particle_cell_mapping.hpp"
#include "particle_mesh_interface.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

/**
 * Class to map particle positions to Nektar++ cells. Implemented for triangles
 * and quads.
 */
class NektarGraphLocalMapperT : public LocalMapper {
private:
  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  const double tol;

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
        particle_mesh_interface(particle_mesh_interface), tol(tol){};

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  triangles and quads.
   */
  inline void map(ParticleGroup &particle_group, const int map_cell = -1) {

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

    for (int cellx = cell_start; cellx < cell_end; cellx++) {
      // for (int cellx = 0; cellx < ncell; cellx++) {

      auto t0_copy_from = profile_timestamp();
      position_dat->cell_dat.get_cell_async(cellx, particle_positions,
                                            event_stack);
      ref_position_dat->cell_dat.get_cell_async(cellx, ref_particle_positions,
                                                event_stack);
      mpi_rank_dat->cell_dat.get_cell_async(cellx, mpi_ranks, event_stack);
      cell_id_dat->cell_dat.get_cell_async(cellx, cell_ids, event_stack);

      event_stack.wait();
      sycl_target->profile_map.inc(
          "NektarGraphLocalMapperT", "copy_from", 0,
          profile_elapsed(t0_copy_from, profile_timestamp()));

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
          NekDouble dist;

          bool geom_found = false;
          // check the original nektar++ geoms
          for (auto &ex : element_ids) {
            Geometry2DSharedPtr geom_2d = graph->GetGeometry2D(ex);
            geom_found = geom_2d->ContainsPoint(global_coord, local_coord,
                                                this->tol, dist);
            if (geom_found) {
              (mpi_ranks)[1][rowx] = rank;
              (cell_ids)[0][rowx] = ex;
              for (int dimx = 0; dimx < ndim; dimx++) {
                ref_particle_positions[dimx][rowx] = local_coord[dimx];
              }
              break;
            }
          }
          sycl_target->profile_map.inc(
              "NektarGraphLocalMapperT", "map_nektar", 0,
              profile_elapsed(t0_nektar_lookup, profile_timestamp()));

          auto t0_halo_lookup = profile_timestamp();
          // containing geom not found in the set of owned geoms, now check the
          // remote geoms
          if (!geom_found) {
            for (auto &remote_geom :
                 this->particle_mesh_interface->remote_triangles) {
              geom_found = remote_geom->geom->ContainsPoint(
                  global_coord, local_coord, this->tol, dist);
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
              geom_found = remote_geom->geom->ContainsPoint(
                  global_coord, local_coord, this->tol, dist);
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
          sycl_target->profile_map.inc(
              "NektarGraphLocalMapperT", "map_halo", 0,
              profile_elapsed(t0_halo_lookup, profile_timestamp()));
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

      sycl_target->profile_map.inc(
          "NektarGraphLocalMapperT", "copy_to", 0,
          profile_elapsed(t0_copy_to, profile_timestamp()));
    }
    sycl_target->profile_map.inc("NektarGraphLocalMapperT", "map", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };
};

} // namespace NESO

#endif
