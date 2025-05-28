#include "nektar_interface/particle_cell_mapping/map_particles_host.hpp"

namespace NESO {

/**
 *  Constructor for mapping class.
 *
 *  @param sycl_target SYCLTarget on which to perform mapping.
 *  @param particle_mesh_interface ParticleMeshInterface containing 2D
 * Nektar++ cells.
 */
MapParticlesHost::MapParticlesHost(
    SYCLTargetSharedPtr sycl_target,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    ParameterStoreSharedPtr config)
    : sycl_target(sycl_target),
      particle_mesh_interface(particle_mesh_interface) {

  this->tol = config->get("MapParticlesHost/tol", 0.0);
}

/**
 *  Called internally by NESO-Particles to map positions to Nektar++
 *  triangles and quads.
 */
void MapParticlesHost::map(ParticleGroup &particle_group, const int map_cell) {

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

      // Is this particle already binned into a cell?
      if ((mpi_ranks)[1][rowx] < 0) {
        if (Debug::enabled(Debug::MOVEMENT_LEVEL)) {
          nprint("MapParticlesHost::map");
          nprint("\tcell:", cellx, "layer:", rowx);
        }

        // copy the particle position into a nektar++ point format
        for (int dimx = 0; dimx < ndim; dimx++) {
          global_coord[dimx] = particle_positions[dimx][rowx];
          local_coord[dimx] = ref_particle_positions[dimx][rowx];
        }

        if (Debug::enabled(Debug::MOVEMENT_LEVEL)) {
          nprint("\tglobal_coord:", global_coord[0], global_coord[1],
                 global_coord[2]);
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
          if (ndim == 2) {
            Geometry2DSharedPtr geom_2d = graph->GetGeometry2D(ex);
            geom_found =
                contains_point_2d(geom_2d, global_coord, local_coord, tol);
          } else if (ndim == 3) {
            Geometry3DSharedPtr geom_3d = get_geometry_3d(graph, ex);
            geom_found =
                contains_point_3d(geom_3d, global_coord, local_coord, tol);
          }
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

        if (ndim == 2) {
          if (!geom_found) {
            for (auto &remote_geom :
                 this->particle_mesh_interface->remote_triangles) {

              geom_found = contains_point_2d(remote_geom->geom, global_coord,
                                             local_coord, tol);
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
                                             local_coord, tol);
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
        } else if (ndim == 3) {
          if (!geom_found) {
            for (auto &remote_geom :
                 this->particle_mesh_interface->remote_geoms_3d) {

              geom_found = contains_point_3d(remote_geom->geom, global_coord,
                                             local_coord, tol);
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
        }

        time_halo_lookup +=
            profile_elapsed(t0_halo_lookup, profile_timestamp());
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

  sycl_target->profile_map.inc("NektarGraphLocalMapper", "copy_to", 0,
                               time_copy_to);
  sycl_target->profile_map.inc("NektarGraphLocalMapper", "map_halo", 0,
                               time_halo_lookup);
  sycl_target->profile_map.inc("NektarGraphLocalMapper", "map_nektar", 0,
                               time_map_nektar);
  sycl_target->profile_map.inc("NektarGraphLocalMapper", "copy_from", 0,
                               time_copy_from);
  sycl_target->profile_map.inc("NektarGraphLocalMapper", "map", 1,
                               profile_elapsed(t0, profile_timestamp()));
}

} // namespace NESO
