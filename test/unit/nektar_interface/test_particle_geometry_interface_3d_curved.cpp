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

TEST(ParticleGeometryInterfaceCurved, Init) {

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
  // std::filesystem::path mesh_file =
  //     test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";

  std::filesystem::path mesh_file =
      "/home/js0259/git-ukaea/NESO-workspace/reference_all_types_cube/"
      "mixed_ref_cube_0.5_perturbed_order_2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);

  // build map from owned mesh hierarchy cells to geoms that touch that cell
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);

  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms;
  get_all_elements_3d(graph, geoms);

  auto lambda_stype = [](auto s) -> std::string {
    switch (s) {
    case eTetrahedron:
      return "Tet";
    case ePyramid:
      return "Pyr";
    case ePrism:
      return "Prism";
    case eHexahedron:
      return "Hex";
    default:
      return "shape unknown";
    }
  };

  auto lambda_btype = [](auto s) -> std::string {
    switch (s) {
    case eModified_A:
      return "eA";
    case eModified_B:
      return "eB";
    case eModified_C:
      return "eC";
    case eModifiedPyr_C:
      return "ePyrC";
    default:
      return "basis unknown";
    }
  };

  for (auto gx : geoms) {
    auto geom = gx.second;
    auto xmap = geom->GetXmap();
    nprint(geom->GetShapeType(), lambda_stype(geom->GetShapeType()));
    nprint("Num bases:", xmap->GetNumBases());
    for (int dx = 0; dx < 3; dx++) {
      nprint("dx:", dx, "type:", lambda_btype(xmap->GetBasisType(dx)),
             xmap->GetBasisNumModes(dx));
    }

    Array<OneD, NekDouble> Lcoord(3);
    Array<OneD, NekDouble> Lcoord2(3);
    Array<OneD, NekDouble> Gcoord(3);
    Array<OneD, NekDouble> Gcoord2(3);
    Lcoord[0] = -0.05;
    Lcoord[1] = -0.05;
    Lcoord[2] = -0.05;

    for (int dx = 0; dx < 3; dx++) {
      // calls phys evaluate which takes a loc coord not a loc collapsed coord
      Gcoord[dx] = geom->GetCoord(dx, Lcoord);
    }

    geom->GetLocCoords(Gcoord, Lcoord2);

    for (int dx = 0; dx < 3; dx++) {
      // calls phys evaluate which takes a loc coord not a loc collapsed coord
      Gcoord2[dx] = geom->GetCoord(dx, Lcoord2);
    }
    nprint(Lcoord[0], Lcoord[1], Lcoord[2], "\n", Lcoord2[0], Lcoord2[1],
           Lcoord2[2], "\n", Gcoord[0], Gcoord[1], Gcoord[2], "\n", Gcoord2[0],
           Gcoord2[1], Gcoord2[2], "\n------");

    auto I0 = xmap->GetBasis(0)->GetI(Lcoord);
    auto I1 = xmap->GetBasis(1)->GetI(Lcoord);
    auto I2 = xmap->GetBasis(2)->GetI(Lcoord);

    for (auto ix = I0->begin(); ix != I0->end(); ix++) {
      nprint(*ix);
    }
    nprint("-----");
    const int npts = xmap->GetTotPoints();
    Array<OneD, NekDouble> ptsx(npts), ptsy(npts), ptsz(npts);
    xmap->BwdTrans(geom->GetCoeffs(0), ptsx);
    xmap->BwdTrans(geom->GetCoeffs(1), ptsy);
    xmap->BwdTrans(geom->GetCoeffs(2), ptsz);

    Array<OneD, NekDouble> eta(3);
    xmap->LocCoordToLocCollapsed(Lcoord, eta);
    Array<OneD, DNekMatSharedPtr> I(3);
    I[0] = xmap->GetBasis(0)->GetI(eta);
    I[1] = xmap->GetBasis(1)->GetI(eta + 1);
    I[2] = xmap->GetBasis(2)->GetI(eta + 2);
    const auto x = xmap->PhysEvaluate(I, ptsx);
    const auto y = xmap->PhysEvaluate(I, ptsy);
    const auto z = xmap->PhysEvaluate(I, ptsz);
    nprint(x, y, z);
  }

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleGeometryInterfaceCurved, XMapNewtonBase) {

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
  // std::filesystem::path mesh_file =
  //     test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";

  std::filesystem::path mesh_file =
      "/home/js0259/git-ukaea/NESO-workspace/reference_all_types_cube/"
      "mixed_ref_cube_0.5_perturbed_order_2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);

  // build map from owned mesh hierarchy cells to geoms that touch that cell
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms;
  get_all_elements_3d(graph, geoms);

  auto lambda_stype = [](auto s) -> std::string {
    switch (s) {
    case eTetrahedron:
      return "Tet";
    case ePyramid:
      return "Pyr";
    case ePrism:
      return "Prism";
    case eHexahedron:
      return "Hex";
    default:
      return "shape unknown";
    }
  };

  auto lambda_btype = [](auto s) -> std::string {
    switch (s) {
    case eModified_A:
      return "eA";
    case eModified_B:
      return "eB";
    case eModified_C:
      return "eC";
    case eModifiedPyr_C:
      return "ePyrC";
    default:
      return "basis unknown";
    }
  };

  auto lambda_forward_map = [&](auto geom, const auto &xi, auto &phys) {
    // Evaluate the forward map from xi to physical space using the expansion.
    auto xmap = geom->GetXmap();
    auto I0 = xmap->GetBasis(0)->GetI(xi);
    auto I1 = xmap->GetBasis(1)->GetI(xi);
    auto I2 = xmap->GetBasis(2)->GetI(xi);
    const int npts = xmap->GetTotPoints();
    Array<OneD, NekDouble> ptsx(npts), ptsy(npts), ptsz(npts);
    xmap->BwdTrans(geom->GetCoeffs(0), ptsx);
    xmap->BwdTrans(geom->GetCoeffs(1), ptsy);
    xmap->BwdTrans(geom->GetCoeffs(2), ptsz);
    Array<OneD, NekDouble> eta(3);
    xmap->LocCoordToLocCollapsed(xi, eta);
    Array<OneD, DNekMatSharedPtr> I(3);
    I[0] = xmap->GetBasis(0)->GetI(eta);
    I[1] = xmap->GetBasis(1)->GetI(eta + 1);
    I[2] = xmap->GetBasis(2)->GetI(eta + 2);
    const auto x = xmap->PhysEvaluate(I, ptsx);
    const auto y = xmap->PhysEvaluate(I, ptsy);
    const auto z = xmap->PhysEvaluate(I, ptsz);
    phys[0] = x;
    phys[1] = y;
    phys[2] = z;
  };

  auto lambda_test_contained_dir = [](const REAL eta) -> REAL {
    if ((eta >= -1.0) && (eta <= 1.0)) {
      return 0.0;
    } else {
      if (eta < -1.0) {
        return -1.0 - eta;
      } else {
        return eta - 1.0;
      }
    }
  };

  auto lambda_test_contained = [&](const auto eta0, const auto eta1,
                                   const auto eta2) -> REAL {
    return std::max(lambda_test_contained_dir(eta0),
                    std::max(lambda_test_contained_dir(eta1),
                             lambda_test_contained_dir(eta2)));
  };

  auto rng = std::mt19937(9457 + rank);
  auto lambda_sample_internal_point = [&](auto geom, Array<OneD, NekDouble> &xi,
                                          Array<OneD, NekDouble> &phys) {
    auto contained = false;
    auto bounding_box = geom->GetBoundingBox();
    std::uniform_real_distribution<double> uniform0(bounding_box[0],
                                                    bounding_box[3]);
    std::uniform_real_distribution<double> uniform1(bounding_box[1],
                                                    bounding_box[4]);
    std::uniform_real_distribution<double> uniform2(bounding_box[2],
                                                    bounding_box[5]);

    INT c = 0;
    while (!contained) {
      phys[0] = uniform0(rng);
      phys[1] = uniform1(rng);
      phys[2] = uniform2(rng);
      contained = geom->ContainsPoint(phys, xi, 1.0e-12);
      if (1e4 < c++) {
        nprint(
            "Test point sampling is taking an excessive number of iterations.");
      }
    }
    lambda_forward_map(geom, xi, phys);
  };

  auto lambda_check_x_map = [&](auto geom, Array<OneD, NekDouble> &xi,
                                Array<OneD, NekDouble> &phys) {
    nprint("CHECK X MAP START", xi[0], xi[1], xi[2], phys[0], phys[1], phys[2]);

    REAL eta0, eta1, eta2;
    const int shape_type_int = static_cast<int>(geom->GetShapeType());
    // The point we sampled was considered contained by Nektar++
    // Check that this alternative reference point is also contained.
    const REAL xi00 = xi[0];
    const REAL xi01 = xi[1];
    const REAL xi02 = xi[2];
    GeometryInterface::loc_coord_to_loc_collapsed_3d(shape_type_int, xi00, xi01,
                                                     xi02, &eta0, &eta1, &eta2);
    const auto dist = lambda_test_contained(eta0, eta1, eta2);
    nprint("dist0", dist, eta0, eta1, eta2);
    EXPECT_TRUE(dist < 1.0e-8);

    // test the forward x map from reference space to physical space
    Newton::XMapNewton<Newton::MappingGeneric3D> mapper(sycl_target, geom);
    REAL test_phys0, test_phys1, test_phys2;
    mapper.x(xi[0], xi[1], xi[2], &test_phys0, &test_phys1, &test_phys2);
    EXPECT_NEAR(phys[0], test_phys0, 1.0e-10);
    EXPECT_NEAR(phys[1], test_phys1, 1.0e-10);
    EXPECT_NEAR(phys[2], test_phys2, 1.0e-10);

    REAL test_xi0, test_xi1, test_xi2;
    const bool converged = mapper.x_inverse(
        phys[0], phys[1], phys[2], &test_xi0, &test_xi1, &test_xi2, 1.0e-10);
    EXPECT_TRUE(converged);

    const bool same_xi = (std::abs(test_xi0 - xi[0]) < 1.0e-8) &&
                         (std::abs(test_xi1 - xi[1]) < 1.0e-8) &&
                         (std::abs(test_xi2 - xi[2]) < 1.0e-8);

    // If the test_xi is xi then the inverse mapping was good
    // Otherwise check test_xi is a reference coordinate that maps to phys
    if (!same_xi) {
      Array<OneD, NekDouble> test_phys(3);
      lambda_forward_map(geom, xi, test_phys);
      const bool equiv_xi = (std::abs(test_phys[0] - phys[0]) < 1.0e-8) &&
                            (std::abs(test_phys[1] - phys[1]) < 1.0e-8) &&
                            (std::abs(test_phys[2] - phys[2]) < 1.0e-8);

      // The point we sampled was considered contained by Nektar++
      // Check that this alternative reference point is also contained.
      const REAL xi00 = test_xi0;
      const REAL xi01 = test_xi1;
      const REAL xi02 = test_xi2;
      GeometryInterface::loc_coord_to_loc_collapsed_3d(
          shape_type_int, xi00, xi01, xi02, &eta0, &eta1, &eta2);

      const auto dist = lambda_test_contained(eta0, eta1, eta2);
      nprint("dist1", dist, eta0, eta1, eta2);
      EXPECT_TRUE(dist < 1.0e-8);
    }

    nprint("CHECK X MAP END");
  };

  for (auto gx : geoms) {
    auto geom = gx.second;
    auto xmap = geom->GetXmap();
    const int shape_type_int = static_cast<int>(geom->GetShapeType());
    nprint(geom->GetShapeType(), lambda_stype(geom->GetShapeType()));
    nprint("Num bases:", xmap->GetNumBases());
    for (int dx = 0; dx < 3; dx++) {
      nprint("dx:", dx, "type:", lambda_btype(xmap->GetBasisType(dx)),
             xmap->GetBasisNumModes(dx));
    }

    Array<OneD, NekDouble> test_xi(3);
    Array<OneD, NekDouble> test_phys(3);
    lambda_sample_internal_point(geom, test_xi, test_phys);
    lambda_check_x_map(geom, test_xi, test_phys);
  }

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
