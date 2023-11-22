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
    const int shape_type_int = static_cast<int>(geom->GetShapeType());
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
    
    
    const REAL xi00 = Lcoord[0];
    const REAL xi01 = Lcoord[1];
    const REAL xi02 = Lcoord[2];
    REAL eta0, eta1, eta2;

    GeometryInterface::loc_coord_to_loc_collapsed_3d(shape_type_int, xi00,
                                          xi01, xi02, &eta0,
                                          &eta1, &eta2);
    EXPECT_NEAR(eta0, eta[0], 1.0e-12);
    EXPECT_NEAR(eta1, eta[1], 1.0e-12);
    EXPECT_NEAR(eta2, eta[2], 1.0e-12);
    nprint("coords:", xi00, xi01, xi02, eta0, eta1, eta2);

    REAL xi0, xi1, xi2;
    const bool converged =
        mapper.x_inverse(phys0, phys1, phys2, &xi0, &xi1, &xi2, 1.0e-10);
    nprint("Q", xi0, xi1, xi2, converged);
    // these might be quite far depending on the map - the residual is on the X
    // map output
    EXPECT_NEAR(xi0, Lcoord[0], 1.0e-2);
    EXPECT_NEAR(xi1, Lcoord[1], 1.0e-2);
    EXPECT_NEAR(xi2, Lcoord[2], 1.0e-2);

    Array<OneD, NekDouble> Lcoordt(3);
    Lcoordt[0] = xi0;
    Lcoordt[1] = xi1;
    Lcoordt[2] = xi2;
    const REAL g0 = geom->GetCoord(0, Lcoordt);
    const REAL g1 = geom->GetCoord(1, Lcoordt);
    const REAL g2 = geom->GetCoord(2, Lcoordt);
    EXPECT_NEAR(g0, Gcoord[0], 1.0e-5);
    EXPECT_NEAR(g1, Gcoord[1], 1.0e-5);
    EXPECT_NEAR(g2, Gcoord[2], 1.0e-5);



    if (static_cast<int>(geom->GetShapeType()) == 5) {
      auto z0 = xmap->GetBase()[0]->GetZ();
      auto bw0 = xmap->GetBase()[0]->GetBaryWeights();
      const int N0 = 4;
      std::vector<REAL> div_space0(N0);
      Bary::preprocess_weights(N0, eta[0], &z0[0], &bw0[0], div_space0.data());
      for (int ix = 0; ix < N0; ix++) {
        nprint("host0:", ix, div_space0[ix]);
      }
      auto z1 = xmap->GetBase()[1]->GetZ();
      auto bw1 = xmap->GetBase()[1]->GetBaryWeights();
      const int N1 = 3;
      std::vector<REAL> div_space1(N1);
      Bary::preprocess_weights(N1, eta[1], &z1[0], &bw1[0], div_space1.data());
      for (int ix = 0; ix < N1; ix++) {
        nprint("host1:", ix, div_space1[ix]);
      }
    }
  }

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
