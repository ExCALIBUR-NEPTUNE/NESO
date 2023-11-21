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
    nprint("P", x, y, z);

    Newton::XMapNewton<Newton::MappingGeneric3D> mapper(sycl_target, geom);

    REAL phys0, phys1, phys2;
    mapper.x(Lcoord[0], Lcoord[1], Lcoord[2], &phys0, &phys1, &phys2);
    nprint("N", phys0, phys1, phys2);

    EXPECT_NEAR(phys0, x, 1.0e-10);
    EXPECT_NEAR(phys1, y, 1.0e-10);
    EXPECT_NEAR(phys2, z, 1.0e-10);

    REAL xi0, xi1, xi2;
    mapper.x_inverse(phys0, phys1, phys2, &xi0, &xi1, &xi2);
    nprint("Q", xi0, xi1, xi2);
  }

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
