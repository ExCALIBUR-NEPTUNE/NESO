#ifndef __GEOMETRY_TRANSPORT_3D_H__
#define __GEOMETRY_TRANSPORT_3D_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>

// System includes
#include <iostream>
#include <mpi.h>

using namespace std;
using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::LibUtilities;

#include "geometry_container_3d.hpp"
#include "geometry_transport_2d.hpp"
#include "remote_geom_3d.hpp"
#include "shape_mapping.hpp"

namespace NESO {

/**
 * Get all 3D geometry objects from a Nektar++ MeshGraph
 *
 * @param[in] graph MeshGraph instance.
 * @param[in,out] std::map of Nektar++ Geometry3D pointers.
 */
void get_all_elements_3d(
    Nektar::SpatialDomains::MeshGraphSharedPtr &graph,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> &geoms);

/**
 * Get a local 3D geometry object from a Nektar++ MeshGraph
 *
 * @param graph Nektar++ MeshGraph to return geometry object from.
 * @returns Local 3D geometry object.
 */
Geometry3DSharedPtr
get_element_3d(Nektar::SpatialDomains::MeshGraphSharedPtr &graph);

/**
 * Categorise geometry types by shape, local or remote and X-map type.
 *
 * @param[in] graph Nektar MeshGraph of locally owned geometry objects.
 * @param[in] remote_geoms_3d Vector of remotely owned geometry objects.
 * @param[in, out] output_container Geometry objects cataorised by type..
 */
void assemble_geometry_container_3d(
    Nektar::SpatialDomains::MeshGraphSharedPtr &graph,
    std::vector<std::shared_ptr<RemoteGeom3D>> &remote_geoms_3d,
    GeometryContainer3D &output_container);

/**
 * Create 3D geometry objects from integer descriptions in the format:
 * <owning rank>, <geometry id>, <faces as list of 2D geometry ids>
 *
 * @param[in,out] base_ptr Pointer to pointer of start of integer array. The
 * array pointer is incremented as the integers are consumed to create 3D
 * geometry objects.
 * @param[in] rank_element_map Map from MPI rank to a map from geometry id to
 * 2D geom shared pointer.
 */
std::shared_ptr<RemoteGeom3D> reconstruct_geom_3d(
    int **base_ptr,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>>
        &rank_element_map_2d);

/**
 * Reconstruct multiple 3D geometry objects from a vector of ints.
 *
 * @param[in] rank_element_map_2d Map from MPI rank to element map (geom id to
 * shared ptr) required to rebuild geoms.
 * @param[in] packed_geoms Int vector of 3D geometry objects described by 2D
 * geometry ids.
 * @param[in,out] output_container Output vector in which to place constructed
 * 3D geometry objects.
 */
void reconstruct_geoms_3d(
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>>
        &rank_element_map_2d,
    std::vector<int> &packed_geoms,
    std::vector<std::shared_ptr<RemoteGeom3D>> &output_container);

/**
 * Deconstruct a 3D geometry object into a series of integers that describe
 * the owning rank and shape type.
 *
 * @param[in] rank MPI rank that owns the 3D object.
 * @param[in] geometry_id Global id of geometry object.
 * @param[in] geom 3D geometry object.
 * @param[in,out] deconstructed_geoms Output vector to push description onto.
 */
void deconstruct_geoms_base_3d(const int rank, const int geometry_id,
                               std::shared_ptr<Geometry3D> geom,
                               std::vector<int> &deconstructed_geoms);

/**
 * Deconstruct a 3D geometry object into a series of integers that describe
 * the owning rank, shape type and constituent faces.
 *
 * @param[in] rank MPI rank that owns the 3D object.
 * @param[in] geometry_id Global id of geometry object.
 * @param[in] geom 3D geometry object.
 * @param[in,out] deconstructed_geoms Output vector to push description onto.
 * @param[in,out] face_ids Set of face ids to add faces to.
 */
void deconstruct_geoms_3d(const int rank, const int geometry_id,
                          std::shared_ptr<Geometry3D> geom,
                          std::vector<int> &deconstructed_geoms,
                          std::set<int> &face_ids);

/**
 *  Deconstruct, i.e. serialise to integers, 3D Nektar++ geometry objects such
 *  that they can be sent over MPI. Geometry objects are packed into an output
 *  map for each remote MPI rank.
 *
 *  @param[in] original_rank MPI rank which owns the geometry objects in
 * `rank_element_map`.
 *  @param[in] geoms_2d Map from geometry id to 2D geometry objects required by
 *  the 3D geometry objects which are t o be packed.
 *  @param[in] rank_element_map Map from remote MPI rank to map from geometry id
 * to 3D geometry object to pack for the remote rank.
 *  @param[in] send_ranks Vector of remote MPI ranks.
 *  @param[in, out] send_sizes Output vector of packed sizes per MPI rank.
 *  @param[in, out] deconstructed_geoms Map from remote MPI rank to vector of
 *  ints describing the packed 3D geoms in terms of the 2D geometry ids that
 *  build the faces of the 3D object.
 *  @param[in, out] rank_triangle_map Map from remote MPI rank to TriGeom ids
 * and objects to pack and send.
 *  @param[in, out] rank_quad Map from remote MPI rank to QuadGeom ids and
 * objects to pack and send.
 */
void deconstuct_per_rank_geoms_3d(
    const int original_rank,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
        &geoms_2d,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>>
        &rank_element_map,
    const int num_send_ranks, std::vector<int> &send_ranks,
    std::vector<int> &send_sizes,
    std::map<int, std::vector<int>> &deconstructed_geoms,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::TriGeom>>>
        &rank_triangle_map,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::QuadGeom>>>
        &rank_quad_map);

/**
 *  Exchange 3D geometry objects between MPI ranks. Must be called collectively
 * on the communicator.
 *
 *  @param[in] comm MPI communicator.
 *  @param[in] deconstructed_geoms Map from remote MPI rank to packed 3D
 * geometry objects to send to that MPI rank.
 *  @param[in] send_ranks Vector of MPI ranks to send 3D geometry objects to.
 *  @param[in] send_sizes The number of ints to send to each remote MPI rank.
 *  @param[in] num_recv_ranks Number of MPI ranks to receive 3D geometry objects
 * from.
 *  @param[in] recv_rank Vector of MPI ranks to receive 3D geometry objects
 * from.
 *  @param[in] recv_sizes Vector containing the number of ints expected to be
 * received from each remote MPI rank.
 *  @param[in, out] packed_geoms Received 3D geometry objects in packed int
 * form.
 */
void sendrecv_geoms_3d(MPI_Comm comm,
                       std::map<int, std::vector<int>> &deconstructed_geoms,
                       const int num_send_ranks, std::vector<int> &send_ranks,
                       std::vector<int> &send_sizes, const int num_recv_ranks,
                       std::vector<int> &recv_ranks,
                       std::vector<int> &recv_sizes,
                       std::vector<int> &packed_geoms);

} // namespace NESO

#endif
