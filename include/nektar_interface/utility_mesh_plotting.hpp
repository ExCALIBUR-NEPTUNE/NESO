#ifndef __UTILITY_MESH_PLOTTING_H_
#define __UTILITY_MESH_PLOTTING_H_

#include "nektar_interface/particle_interface.hpp"
#include <SpatialDomains/Geometry.h>
#include <SpatialDomains/Geometry2D.h>
#include <SpatialDomains/Geometry3D.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <neso_particles.hpp>
#include <string>
#include <vector>

using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

namespace NESO {

/**
 *  Class to write Nektar++ geometry objects to a vtk file as a collection of
 *  vertices and edges for visualisation in paraview.
 */
class VTKGeometryWriter {
protected:
  std::vector<std::shared_ptr<Geometry>> geoms;
  std::map<std::shared_ptr<Geometry>, REAL> geom_props;

  inline REAL get_property(const std::shared_ptr<Geometry> ptr) {
    if (this->geom_props.count(ptr)) {
      return this->geom_props.at(ptr);
    } else {
      return 0.0;
    }
  }

public:
  int rank;

  VTKGeometryWriter() : rank(0) {};
  VTKGeometryWriter(const int rank) : rank(rank) {};

  /**
   *  Push a geometry object onto the collection of objects to write to a vtk
   * file along with a REAL property.
   *
   *  @param[in] geom Shared pointer to Nektar++ geometry object.
   *  @param[in] prop Property to write with geometry object.
   */
  template <typename T>
  inline void push_back(std::shared_ptr<T> &geom, const REAL prop) {
    auto ptr = std::dynamic_pointer_cast<Geometry>(geom);
    this->geoms.push_back(ptr);
    this->geom_props[ptr] = prop;
  }

  /**
   *  Push a geometry object onto the collection of objects to write to a vtk
   * file.
   *
   *  @param[in] geom Shared pointer to Nektar++ geometry object.
   */
  template <typename T> inline void push_back(std::shared_ptr<T> &geom) {
    this->push_back(geom, 0.0);
  }

  /**
   *  Write vtk file to disk with given filename. Filename should end with .vtk.
   *
   *  @param filename[in] Input filename.
   */
  inline void write(std::string filename) {
    // vertices required
    std::map<int, std::shared_ptr<PointGeom>> vertices;
    // edges required
    std::map<int, std::shared_ptr<Geometry1D>> edges;

    for (auto &geom : this->geoms) {
      const int num_edges = geom->GetNumEdges();
      for (int edgex = 0; edgex < num_edges; edgex++) {
        Geometry1DSharedPtr edge = geom->GetEdge(edgex);
        const int num_verts = edge->GetNumVerts();
        NESOASSERT(num_verts == 2, "Expected edge to only have 2 vertices");
        for (int vertexx = 0; vertexx < num_verts; vertexx++) {
          PointGeomSharedPtr vertex = edge->GetVertex(vertexx);
          const int gid = vertex->GetGlobalID();
          vertices[gid] = vertex;
        }
        const int gid = edge->GetGlobalID();
        edges[gid] = edge;
      }
    }

    std::ofstream vtk_file;
    vtk_file.open(filename);

    vtk_file << "# vtk DataFile Version 2.0\n";
    vtk_file << "NESO Unstructured Grid\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n\n";
    vtk_file << "POINTS " << vertices.size() << " float\n";

    // map from nektar++ geometry id to vtk vertex id
    std::map<int, int> gid_to_vid;
    int next_vid = 0;
    for (auto &vertex : vertices) {
      const int gid = vertex.first;
      gid_to_vid[gid] = next_vid;
      Nektar::NekDouble x, y, z;
      vertex.second->GetCoords(x, y, z);
      const int coordim = vertex.second->GetCoordim();

      if (coordim == 1) {
        vtk_file << x << " " << 0.0 << " " << 0.0 << " \n";
      } else if (coordim == 2) {
        vtk_file << x << " " << y << " " << 0.0 << " \n";
      } else {
        vtk_file << x << " " << y << " " << z << " \n";
      }

      next_vid++;
    }
    vtk_file << "\n";

    std::map<LibUtilities::ShapeType, int> map_shape_to_vtk;
    map_shape_to_vtk[ePoint] = 1;
    map_shape_to_vtk[eSegment] = 3;
    map_shape_to_vtk[eTriangle] = 5;
    map_shape_to_vtk[eQuadrilateral] = 9;
    map_shape_to_vtk[eTetrahedron] = 10;
    map_shape_to_vtk[eHexahedron] = 12;
    map_shape_to_vtk[ePrism] = 13;
    map_shape_to_vtk[ePyramid] = 14;

    std::vector<int> cell_ints;
    std::vector<int> cell_type_ints;
    std::vector<REAL> properties;
    properties.reserve(this->geoms.size() + edges.size());

    for (auto &edge : edges) {
      cell_ints.push_back(2);
      cell_ints.push_back(
          gid_to_vid.at(edge.second->GetVertex(0)->GetGlobalID()));
      cell_ints.push_back(
          gid_to_vid.at(edge.second->GetVertex(1)->GetGlobalID()));
      cell_type_ints.push_back(3);
      properties.push_back(0.0);
    }

    for (auto &geom : this->geoms) {
      // Not sure if the vertex ordering of Nektar matches VTK so restrict to
      // 2D for now.
      if (geom->GetCoordim() == 2) {
        const int num_verts = geom->GetNumVerts();
        cell_ints.push_back(num_verts);
        for (int vx = 0; vx < num_verts; vx++) {
          auto vertex = geom->GetVertex(vx);
          const int vertex_gid = vertex->GetVid();
          const int vertex_vtk_id = gid_to_vid[vertex_gid];
          cell_ints.push_back(vertex_vtk_id);
        }
        cell_type_ints.push_back(map_shape_to_vtk.at(geom->GetShapeType()));
      }
      properties.push_back(this->get_property(geom));
    }

    const int num_objs = cell_type_ints.size();
    vtk_file << "CELLS " << num_objs << " " << cell_ints.size() << "\n";
    for (const int ix : cell_ints) {
      vtk_file << ix << " ";
    }
    vtk_file << "\n";
    vtk_file << "\n";
    vtk_file << "CELL_TYPES " << num_objs << "\n";
    for (int ix = 0; ix < num_objs; ix++) {
      vtk_file << cell_type_ints.at(ix) << " ";
    }

    vtk_file << "\n";
    vtk_file << "CELL_DATA " << num_objs << "\n";
    vtk_file << "SCALARS rank float 1\n";
    vtk_file << "LOOKUP_TABLE CellColors\n";
    for (int gx = 0; gx < num_objs; gx++) {
      vtk_file << this->rank << " ";
    }

    vtk_file << "\n";
    vtk_file << "SCALARS generic_property float 1\n";
    vtk_file << "LOOKUP_TABLE CellColors\n";
    for (int gx = 0; gx < num_objs; gx++) {
      vtk_file << properties.at(gx) << " ";
    }

    vtk_file.close();
  }
};

/**
 * Write the vertices and edges of the owned geometry objects on this rank to a
 * vtk file.
 *
 * @param[in] filename Filename to write .<rank>.vtk will be appended to the
 * given filename.
 * @param[in] particle_mesh_interface ParticleMeshInterface containing geometry
 * objects.
 */
inline void
write_vtk_cells_owned(std::string filename,
                      ParticleMeshInterfaceSharedPtr particle_mesh_interface) {

  const int ndim = particle_mesh_interface->ndim;
  const int rank = particle_mesh_interface->comm_rank;
  filename += "." + std::to_string(rank) + ".vtk";

  VTKGeometryWriter vtk_writer{rank};

  if (ndim == 2) {
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>> geoms;
    get_all_elements_2d(particle_mesh_interface->graph, geoms);
    for (auto &geom : geoms) {
      vtk_writer.push_back(geom.second);
    }
  } else if (ndim == 3) {
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms;
    get_all_elements_3d(particle_mesh_interface->graph, geoms);
    for (auto &geom : geoms) {
      vtk_writer.push_back(geom.second);
    }
  }

  vtk_writer.write(filename);
}

/**
 * Write the vertices and edges of the halo geometry objects on this rank to a
 * vtk file.
 *
 * @param[in] filename Filename to write .<rank>.vtk will be appended to the
 * given filename.
 * @param[in] particle_mesh_interface ParticleMeshInterface containing geometry
 * objects.
 */
inline void
write_vtk_cells_halo(std::string filename,
                     ParticleMeshInterfaceSharedPtr particle_mesh_interface) {

  const int ndim = particle_mesh_interface->ndim;
  const int rank = particle_mesh_interface->comm_rank;
  filename += "." + std::to_string(rank) + ".vtk";

  VTKGeometryWriter vtk_writer{rank};

  if (ndim == 2) {
    for (auto &geom : particle_mesh_interface->remote_triangles) {
      vtk_writer.push_back(geom->geom, geom->rank);
    }
    for (auto &geom : particle_mesh_interface->remote_quads) {
      vtk_writer.push_back(geom->geom, geom->rank);
    }
  } else if (ndim == 3) {
    for (auto &geom : particle_mesh_interface->remote_geoms_3d) {
      vtk_writer.push_back(geom->geom, geom->rank);
    }
  }

  vtk_writer.write(filename);
}

/**
 * Write the vertices and edges of the owned mesh hierarchy cells on this rank
 * to a vtk file.
 *
 * @param[in] filename Filename to write .<rank>.vtk will be appended to the
 * given filename.
 * @param[in] particle_mesh_interface ParticleMeshInterface containing geometry
 * objects.
 */
inline void write_vtk_mesh_hierarchy_cells_owned(
    std::string filename,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface) {
  const int rank = particle_mesh_interface->comm_rank;
  filename += "." + std::to_string(rank) + ".vtk";
  VTKMeshHierarchyCellsWriter mh_writer(
      particle_mesh_interface->mesh_hierarchy);

  for (auto cell : particle_mesh_interface->owned_mh_cells) {
    mh_writer.push_back(cell);
  }

  mh_writer.write(filename);
}

/**
 * Write the vertices and edges of all the fine mesh hierarchy cells to a file.
 * Only write from rank 0.
 *
 * @param[in] filename Filename to write to, should end in .vtk.
 * @param[in] particle_mesh_interface ParticleMeshInterface containing geometry
 * objects.
 */
inline void write_vtk_mesh_hierarchy_cells_fine(
    std::string filename,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface) {
  VTKMeshHierarchyCellsWriter mh_writer(
      particle_mesh_interface->mesh_hierarchy);
  mh_writer.write_all_fine(filename);
}

/**
 * Write the vertices and edges of all the coarse mesh hierarchy cells to a
 * file. Only write from rank 0.
 *
 * @param[in] filename Filename to write to, should end in .vtk.
 * @param[in] particle_mesh_interface ParticleMeshInterface containing geometry
 * objects.
 */
inline void write_vtk_mesh_hierarchy_cells_coarse(
    std::string filename,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface) {
  VTKMeshHierarchyCellsWriter mh_writer(
      particle_mesh_interface->mesh_hierarchy);
  mh_writer.write_all_coarse(filename);
}

} // namespace NESO

#endif
