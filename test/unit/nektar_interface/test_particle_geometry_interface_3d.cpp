#include "nektar_interface/coordinate_mapping.hpp"
#include "nektar_interface/geometry_transport/halo_extension.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utility_mesh_plotting.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <neso_particles.hpp>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::LibUtilities;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

TEST(ParticleGeometryInterface, ShapeEnums) {

  const int int_tmp0 = shape_type_to_int(ShapeType::eHexahedron);
  ASSERT_EQ(ShapeType::eHexahedron, int_to_shape_type(int_tmp0));
  const int int_tmp1 = shape_type_to_int(ShapeType::ePrism);
  ASSERT_EQ(ShapeType::ePrism, int_to_shape_type(int_tmp1));
  const int int_tmp2 = shape_type_to_int(ShapeType::ePyramid);
  ASSERT_EQ(ShapeType::ePyramid, int_to_shape_type(int_tmp2));
  const int int_tmp3 = shape_type_to_int(ShapeType::eTetrahedron);
  ASSERT_EQ(ShapeType::eTetrahedron, int_to_shape_type(int_tmp3));
  const int int_tmp4 = shape_type_to_int(ShapeType::eQuadrilateral);
  ASSERT_EQ(ShapeType::eQuadrilateral, int_to_shape_type(int_tmp4));
  const int int_tmp5 = shape_type_to_int(ShapeType::eTriangle);
  ASSERT_EQ(ShapeType::eTriangle, int_to_shape_type(int_tmp5));
  std::array<int, 6> int_values = {int_tmp0, int_tmp1, int_tmp2,
                                   int_tmp3, int_tmp4, int_tmp5};
  for (int vx = 0; vx < 6; vx++) {
    for (int ux = vx + 1; ux < 6; ux++) {
      ASSERT_NE(int_values[vx], int_values[ux]);
    }
  }
}

TEST(ParticleGeometryInterface, HaloExtend3D) {
  const int width = 1;

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path conditions_file =
      test_resources_dir / "reference_all_types_cube/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);

  // build map from owned mesh hierarchy cells to geoms that touch that cell
  auto particle_mesh_interface = std::make_shared<ParticleMeshInterface>(graph);

  // write_vtk_cells_owned("3d_owned", particle_mesh_interface);
  // write_vtk_cells_halo("3d_halo", particle_mesh_interface);
  // write_vtk_mesh_hierarchy_cells_owned("3d_mh", particle_mesh_interface);

  std::set<INT> mh_cell_set;
  for (const INT cell : particle_mesh_interface->owned_mh_cells) {
    mh_cell_set.insert(cell);
  }
  std::map<int,
           std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>>
      rank_geoms_3d_map_local;
  std::map<INT, std::vector<std::pair<int, int>>> cells_to_rank_geoms;
  halo_get_rank_to_geoms_3d(particle_mesh_interface, rank_geoms_3d_map_local);
  halo_get_cells_to_geoms_map(particle_mesh_interface, rank_geoms_3d_map_local,
                              mh_cell_set, cells_to_rank_geoms);

  // represent the map from mesh hierarchy cells to geom ids as a matrix where
  // each row is a mesh hierarchy cell
  std::size_t max_size_t = 0;
  for (auto cx : cells_to_rank_geoms) {
    max_size_t = std::max(max_size_t, cx.second.size());
  }
  int max_size_local = static_cast<int>(max_size_t);
  int max_size;
  MPICHK(MPI_Allreduce(&max_size_local, &max_size, 1, MPI_INT, MPI_MAX,
                       MPI_COMM_WORLD));

  const INT ncells_global =
      particle_mesh_interface->mesh_hierarchy->ncells_global;
  const int num_entries = max_size * ncells_global;
  std::vector<int> map_geom_ids(num_entries);
  std::vector<int> local_map_geom_ids(num_entries);
  // write a null value we can identify
  for (int cx = 0; cx < num_entries; cx++) {
    local_map_geom_ids[cx] = -1;
  }
  // populate the map with the owned geoms
  for (auto cell_geoms : cells_to_rank_geoms) {
    const INT cell = cell_geoms.first;
    const auto rank_geoms = cell_geoms.second;

    int index = cell * max_size;
    for (auto rank_geom : rank_geoms) {
      local_map_geom_ids[index++] = rank_geom.second;
    }
  }

  // reduce the map across all ranks using max to create a copy on all ranks
  MPICHK(MPI_Allreduce(local_map_geom_ids.data(), map_geom_ids.data(),
                       num_entries, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

  // check the entries this rank added are untouched
  for (auto cell_geoms : cells_to_rank_geoms) {
    const INT cell = cell_geoms.first;
    const auto rank_geoms = cell_geoms.second;

    int index = cell * max_size;
    for (auto rank_geom : rank_geoms) {
      const int map_val = map_geom_ids[index++];
      ASSERT_EQ(rank_geom.second, map_val);
    }
  }

  extend_halos_fixed_offset(width, particle_mesh_interface);
  // this rank should now hold all the geoms that touch all the mh cells we
  // claimed as well as mh cells we own

  std::set<int> geoms_to_hold;

  std::set<INT> expected_mh_cells;
  halo_get_mesh_hierarchy_cells(width, particle_mesh_interface,
                                expected_mh_cells);

  // push geoms we should have onto a set
  for (const INT cell : expected_mh_cells) {
    for (INT rowx = 0; rowx < max_size; rowx++) {
      const int gid = map_geom_ids[cell * max_size + rowx];
      if (gid > -1) {
        geoms_to_hold.insert(gid);
      }
    }
  }

  auto lambda_remove_gid = [&](const int gid) {
    if (geoms_to_hold.count(gid)) {
      geoms_to_hold.erase(geoms_to_hold.find(gid));
    }
  };

  // loop over remote geoms and remove from set
  for (auto gid_geom : particle_mesh_interface->remote_geoms_3d) {
    lambda_remove_gid(gid_geom->id);
  }

  // loop over owned geoms and remove from set
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
      geoms_3d_local;
  get_all_elements_3d(particle_mesh_interface->graph, geoms_3d_local);
  for (auto gid_geom : geoms_3d_local) {
    lambda_remove_gid(gid_geom.first);
  }

  // assert set is empty
  ASSERT_EQ(geoms_to_hold.size(), 0);

  particle_mesh_interface->free();
  delete[] argv[0];
  delete[] argv[1];
}

TEST(ParticleGeometryInterface, CoordinateMapping3D) {

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path conditions_file =
      test_resources_dir / "reference_all_types_cube/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);

  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms;
  get_all_elements_3d(graph, geoms);

  std::uniform_real_distribution<double> uniform_rng(-1.0, 1.0);
  auto rng = std::mt19937();
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  BufferDeviceHost<double> dh_eta(sycl_target, 3);
  BufferDeviceHost<double> dh_xi(sycl_target, 3);

  for (auto &geom_pair : geoms) {

    auto geom = geom_pair.second;

    Array<OneD, NekDouble> xi0(3);
    Array<OneD, NekDouble> eta0(3);
    double xi1[3] = {0.0, 0.0, 0.0};
    double eta1[3] = {0.0, 0.0, 0.0};

    const double k_eta0 = uniform_rng(rng);
    const double k_eta1 = uniform_rng(rng);
    const double k_eta2 = uniform_rng(rng);

    eta0[0] = k_eta0;
    eta0[1] = k_eta1;
    eta0[2] = k_eta2;
    eta1[0] = k_eta0;
    eta1[1] = k_eta1;
    eta1[2] = k_eta2;

    auto lambda_test_eta = [&]() {
      ASSERT_NEAR(eta0[0], eta1[0], 1.0e-8);
      ASSERT_NEAR(eta0[1], eta1[1], 1.0e-8);
      ASSERT_NEAR(eta0[2], eta1[2], 1.0e-8);
    };

    auto lambda_test_xi = [&]() {
      ASSERT_NEAR(xi0[0], xi1[0], 1.0e-8);
      ASSERT_NEAR(xi0[1], xi1[1], 1.0e-8);
      ASSERT_NEAR(xi0[2], xi1[2], 1.0e-8);
    };

    auto shape_type = geom->GetShapeType();
    const int k_shape_type_int = shape_type_to_int(shape_type);
    if (shape_type == LibUtilities::eTetrahedron) {
      GeometryInterface::Tetrahedron geom_test{};
      geom->GetXmap()->LocCollapsedToLocCoord(eta0, xi0);
      geom_test.loc_collapsed_to_loc_coord(eta1, xi1);
      lambda_test_xi();
      geom->GetXmap()->LocCoordToLocCollapsed(xi0, eta0);
      geom_test.loc_coord_to_loc_collapsed(xi1, eta1);
      lambda_test_eta();
    } else if (shape_type == LibUtilities::ePyramid) {
      GeometryInterface::Pyramid geom_test{};
      geom->GetXmap()->LocCollapsedToLocCoord(eta0, xi0);
      geom_test.loc_collapsed_to_loc_coord(eta1, xi1);
      lambda_test_xi();
      geom->GetXmap()->LocCoordToLocCollapsed(xi0, eta0);
      geom_test.loc_coord_to_loc_collapsed(xi1, eta1);
      lambda_test_eta();
    } else if (shape_type == LibUtilities::ePrism) {
      GeometryInterface::Prism geom_test{};
      geom->GetXmap()->LocCollapsedToLocCoord(eta0, xi0);
      geom_test.loc_collapsed_to_loc_coord(eta1, xi1);
      lambda_test_xi();
      geom->GetXmap()->LocCoordToLocCollapsed(xi0, eta0);
      geom_test.loc_coord_to_loc_collapsed(xi1, eta1);
      lambda_test_eta();
    } else if (shape_type == LibUtilities::eHexahedron) {
      GeometryInterface::Hexahedron geom_test{};
      geom->GetXmap()->LocCollapsedToLocCoord(eta0, xi0);
      geom_test.loc_collapsed_to_loc_coord(eta1, xi1);
      lambda_test_xi();
      geom->GetXmap()->LocCoordToLocCollapsed(xi0, eta0);
      geom_test.loc_coord_to_loc_collapsed(xi1, eta1);
      lambda_test_eta();
    } else {
      // Unknown shape type
      ASSERT_TRUE(false);
    }

    // test the function that maps all types
    dh_eta.h_buffer.ptr[0] = k_eta0;
    dh_eta.h_buffer.ptr[1] = k_eta1;
    dh_eta.h_buffer.ptr[2] = k_eta2;
    dh_eta.host_to_device();
    dh_xi.h_buffer.ptr[0] = -1.0;
    dh_xi.h_buffer.ptr[1] = -1.0;
    dh_xi.h_buffer.ptr[2] = -1.0;
    dh_xi.host_to_device();

    auto k_eta = dh_eta.d_buffer.ptr;
    auto k_xi = dh_xi.d_buffer.ptr;

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task<>([=]() {
            sycl::global_ptr<const double> p_eta{k_eta};
            sycl::vec<double, 3> v_eta{};
            v_eta.load(0, p_eta);
            sycl::vec<double, 3> v_xi{0.0};

            GeometryInterface::loc_collapsed_to_loc_coord(k_shape_type_int,
                                                          v_eta, v_xi);

            sycl::global_ptr<double> p_xi{k_xi};
            v_xi.store(0, p_xi);
          });
        })
        .wait_and_throw();

    dh_xi.device_to_host();
    for (int dimx = 0; dimx < 3; dimx++) {
      xi1[dimx] = dh_xi.h_buffer.ptr[dimx];
      eta0[dimx] = dh_eta.h_buffer.ptr[dimx];
    }
    geom->GetXmap()->LocCollapsedToLocCoord(eta0, xi0);
    lambda_test_xi();

    // test the xi to eta map
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task<>([=]() {
            sycl::global_ptr<const double> p_xi{k_xi};
            sycl::global_ptr<double> p_eta{k_eta};
            sycl::vec<double, 3> v_eta{0.0};
            sycl::vec<double, 3> v_xi{0.0};
            v_xi.load(0, p_xi);

            GeometryInterface::loc_coord_to_loc_collapsed_3d(k_shape_type_int,
                                                             v_xi, v_eta);

            v_eta.store(0, p_eta);
          });
        })
        .wait_and_throw();

    dh_eta.device_to_host();
    for (int dimx = 0; dimx < 3; dimx++) {
      eta1[dimx] = dh_eta.h_buffer.ptr[dimx];
      xi0[dimx] = dh_xi.h_buffer.ptr[dimx];
    }
    geom->GetXmap()->LocCoordToLocCollapsed(xi0, eta0);
    lambda_test_eta();
  }

  delete[] argv[0];
  delete[] argv[1];
}

TEST(ParticleGeometryInterface, PointInSubDomain3D) {

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path conditions_file =
      test_resources_dir / "reference_all_types_cube/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);

  double point[3];
  mesh->get_point_in_subdomain(point);
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms;
  get_all_elements_3d(graph, geoms);

  bool found = false;
  Array<OneD, NekDouble> to_test(3);
  for (int dimx = 0; dimx < 3; dimx++) {
    to_test[dimx] = point[dimx];
  }

  for (auto geom : geoms) {
    if (geom.second->ContainsPoint(to_test)) {
      found = true;
      break;
    }
  }

  ASSERT_TRUE(found);
  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
