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

// TODO REMOVE START
namespace NESO::Newton {

template <typename NEWTON_TYPE> class XMapNewtonTest : public XMapNewton<NEWTON_TYPE> {
public:

  template <typename TYPE_GEOM>
  XMapNewtonTest(SYCLTargetSharedPtr sycl_target, std::shared_ptr<TYPE_GEOM> geom,
             const int num_modes_factor = 1)
      : XMapNewton<NEWTON_TYPE>(sycl_target, geom, num_modes_factor)
  {}

  inline bool x_inverse(const REAL phys0, const REAL phys1, const REAL phys2,
                        REAL *xi0, REAL *xi1, REAL *xi2,
                        const REAL tol = 1.0e-10,
                        const REAL contained_tol = 1.0e-10) {

    const int k_max_iterations = 51;
    auto k_map_data = this->dh_data->d_buffer.ptr;
    auto k_fdata = this->dh_fdata->d_buffer.ptr;
    const REAL k_tol = tol;
    const double k_contained_tol = contained_tol;
    const std::size_t num_bytes_local = this->num_bytes_local;

    const int k_ndim = this->ndim;
    const int grid_size = this->num_modes_factor * this->num_modes;
    const int k_grid_size_x = std::max(grid_size - 1, 1);
    const int k_grid_size_y = k_ndim > 1 ? k_grid_size_x : 1;
    const int k_grid_size_z = k_ndim > 2 ? k_grid_size_x : 1;
    const REAL k_grid_width = 2.0 / (k_grid_size_x);

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor<unsigned char, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              local_mem(sycl::range<1>(num_bytes_local), cgh);
          cgh.parallel_for<>(
              sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
              [=](auto idx) {
                printf("NEWTON TEST ETA GRID:\n");
                MappingNewtonIterationBase<NEWTON_TYPE> k_newton_type{};

                const REAL p0 = phys0;
                const REAL p1 = phys1;
                const REAL p2 = phys2;
                REAL k_xi0;
                REAL k_xi1;
                REAL k_xi2;
                REAL residual;
                bool cell_found = false;

                for (int g2 = 0; (g2 <= k_grid_size_z) && (!cell_found); g2++) {
                  for (int g1 = 0; (g1 <= k_grid_size_y) && (!cell_found);
                       g1++) {
                    for (int g0 = 0; (g0 <= k_grid_size_x) && (!cell_found);
                         g0++) {

                      REAL eta0, eta1, eta2;

                      eta0 = -1.0 + g0 * k_grid_width;
                      eta1 = -1.0 + g1 * k_grid_width;
                      eta2 = -1.0 + g2 * k_grid_width;

                      k_newton_type.loc_collapsed_to_loc_coord(
                          k_map_data, eta0, eta1, eta2, &k_xi0, &k_xi1, &k_xi2);

                      nprint("~~~~~~~~~~~ ETA:", eta0, eta1, eta2, "XI:", k_xi0, k_xi1, k_xi2);

                      // k_newton_type.set_initial_iteration(k_map_data, p0, p1,
                      // p2,
                      //                                     &k_xi0, &k_xi1,
                      //                                     &k_xi2);

                      // Start of Newton iteration
                      REAL xin0, xin1, xin2;
                      REAL f0, f1, f2;

                      residual = k_newton_type.newton_residual(
                          k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, &f0, &f1,
                          &f2, &local_mem[0]);

                      bool diverged = false;
                      bool converged = false;
                      printf("residual: %f\n", residual);
                      for (int stepx = 0; ((stepx < k_max_iterations) &&
                                           (!converged) && (!diverged));
                           stepx++) {
                        printf("STEPX: %d, RES: %16.8e\n", stepx, residual);

                        k_newton_type.newton_step(
                            k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, f0, f1,
                            f2, &xin0, &xin1, &xin2, &local_mem[0]);

                        k_xi0 = xin0;
                        k_xi1 = xin1;
                        k_xi2 = xin2;

                        residual = k_newton_type.newton_residual(
                            k_map_data, k_xi0, k_xi1, k_xi2, p0, p1, p2, &f0,
                            &f1, &f2, &local_mem[0]);

                        diverged = (ABS(k_xi0) > 15.0) || (ABS(k_xi1) > 15.0) ||
                                   (ABS(k_xi2) > 15.0) || (residual != residual);
                        converged = (residual <= k_tol) && (!diverged);
                      }

                      k_newton_type.loc_coord_to_loc_collapsed(
                          k_map_data, k_xi0, k_xi1, k_xi2, &eta0, &eta1, &eta2);

                      // bool contained = ((-1.0 - k_contained_tol) <= eta0) &&
                      //                  (eta0 <= (1.0 + k_contained_tol)) &&
                      //                  ((-1.0 - k_contained_tol) <= eta1) &&
                      //                  (eta1 <= (1.0 + k_contained_tol)) &&
                      //                  ((-1.0 - k_contained_tol) <= eta2) &&
                      //                  (eta2 <= (1.0 + k_contained_tol));

                      nprint("CLAMPED START");
                      nprint("BEFORE CLAMP:", eta0, eta1, eta2);
                      REAL clamped_eta0 = Kernel::min(eta0, 1.0 + k_contained_tol);
                      REAL clamped_eta1 = Kernel::min(eta1, 1.0 + k_contained_tol);
                      REAL clamped_eta2 = Kernel::min(eta2, 1.0 + k_contained_tol);
                      clamped_eta0 = Kernel::max(clamped_eta0, -1.0 - k_contained_tol);
                      clamped_eta1 = Kernel::max(clamped_eta1, -1.0 - k_contained_tol);
                      clamped_eta2 = Kernel::max(clamped_eta2, -1.0 - k_contained_tol);
                      nprint("AFTER  CLAMP:", clamped_eta0, clamped_eta1, clamped_eta2);

                      REAL clamped_xi0, clamped_xi1, clamped_xi2;
                      k_newton_type.loc_collapsed_to_loc_coord(
                          k_map_data, clamped_eta0, clamped_eta1, clamped_eta2, 
                          &clamped_xi0, &clamped_xi1, &clamped_xi2);
                      
                      const REAL clamped_residual = k_newton_type.newton_residual(
                          k_map_data, clamped_xi0, clamped_xi1, clamped_xi2, 
                          p0, p1, p2, &f0, &f1,
                          &f2, &local_mem[0]);

                      const bool contained = clamped_residual <= k_tol;

                      nprint("CLAMPED END");



                      cell_found = contained && converged;
                    }
                  }
                }
                k_fdata[0] = k_xi0;
                k_fdata[1] = k_xi1;
                k_fdata[2] = k_xi2;
                k_fdata[3] = (residual <= tol) ? 1 : -1;
              });
        })
        .wait_and_throw();

    this->dh_fdata->device_to_host();
    *xi0 = this->dh_fdata->h_buffer.ptr[0];
    *xi1 = this->dh_fdata->h_buffer.ptr[1];
    *xi2 = this->dh_fdata->h_buffer.ptr[2];
    return (this->dh_fdata->h_buffer.ptr[3] > 0);
  }

};




}
// TODO REMOVE END




TEST(ParticleGeometryInterfaceCurved, CoordinateMapping) {

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

  // TODO
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

  // TODO clenaup
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

  // TODO cleanip
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

  auto lambda_to_collapsed_coord = [&](
    auto geom,
    auto xi0,
    auto xi1,
    auto xi2,
    auto * eta0,
    auto * eta1,
    auto * eta2
  ){
    auto s = geom->GetShapeType();
    switch (s) {
    case eTetrahedron:
      GeometryInterface::Tetrahedron{}.loc_coord_to_loc_collapsed(xi0, xi1, xi2, eta0, eta1, eta2);
      break;
    case ePyramid:
      GeometryInterface::Pyramid{}.loc_coord_to_loc_collapsed(xi0, xi1, xi2, eta0, eta1, eta2);
      break;
    case ePrism:
      GeometryInterface::Prism{}.loc_coord_to_loc_collapsed(xi0, xi1, xi2, eta0, eta1, eta2);
      break;
    case eHexahedron:
      GeometryInterface::Hexahedron{}.loc_coord_to_loc_collapsed(xi0, xi1, xi2, eta0, eta1, eta2);
      break;
    default:
      return "shape unknown";
    }
  };
  auto lambda_to_local_coord = [&](
    auto geom,
    auto eta0,
    auto eta1,
    auto eta2,
    auto * xi0,
    auto * xi1,
    auto * xi2
  ){
    auto s = geom->GetShapeType();
    switch (s) {
    case eTetrahedron:
      GeometryInterface::Tetrahedron{}.loc_collapsed_to_loc_coord(eta0, eta1, eta2, xi0, xi1, xi2);
      break;
    case ePyramid:
      GeometryInterface::Pyramid{}.loc_collapsed_to_loc_coord(eta0, eta1, eta2, xi0, xi1, xi2);
      break;
    case ePrism:
      GeometryInterface::Prism{}.loc_collapsed_to_loc_coord(eta0, eta1, eta2, xi0, xi1, xi2);
      break;
    case eHexahedron:
      GeometryInterface::Hexahedron{}.loc_collapsed_to_loc_coord(eta0, eta1, eta2, xi0, xi1, xi2);
      break;
    default:
      return "shape unknown";
    }
  };


  for (auto gx : geoms) {
    auto geom = gx.second;
    auto xmap = geom->GetXmap();
    const int shape_type_int = geom->GetShapeType();

    std::set<std::array<int, 3>> vertices_found;
    std::vector<std::vector<REAL>> vertices = {
      {-1.0, -1.0, -1.0},
      { 1.0, -1.0, -1.0},
      {-1.0,  1.0, -1.0},
      { 1.0,  1.0, -1.0},
      {-1.0, -1.0,  1.0},
      { 1.0, -1.0,  1.0},
      {-1.0,  1.0,  1.0},
      { 1.0,  1.0,  1.0}
    };   

    Array<OneD, NekDouble> test_eta(3);
    Array<OneD, NekDouble> test_eta_neso(3);
    Array<OneD, NekDouble> test_xi(3);
    Array<OneD, NekDouble> test_xi_neso(3);
    REAL eta0, eta1, eta2, xi0, xi1, xi2;
    for(auto vx : vertices){
      test_eta[0] = vx.at(0);
      test_eta[1] = vx.at(1);
      test_eta[2] = vx.at(2);
      xmap->LocCollapsedToLocCoord(test_eta, test_xi);
      xmap->LocCoordToLocCollapsed(test_xi, test_eta);
      xmap->LocCollapsedToLocCoord(test_eta, test_xi);
      xmap->LocCoordToLocCollapsed(test_xi, test_eta);
      
      vertices_found.insert(
        {
          (int) std::round(test_eta[0]),
          (int) std::round(test_eta[1]),
          (int) std::round(test_eta[2])
        }
      );
    }

    const int num_vertices_expected = geom->GetNumVerts();
    const int num_vertices_found = vertices_found.size();

    // At the time of writing TetGeom mapping is inconsistent with itself in Nektar++
    const bool nektar_mapping_consistent = num_vertices_found == num_vertices_expected;

    vertices_found.clear();

    for(auto vx : vertices){
      test_eta[0] = vx.at(0);
      test_eta[1] = vx.at(1);
      test_eta[2] = vx.at(2);
      
      GeometryInterface::loc_collapsed_to_loc_coord(shape_type_int, test_eta, test_xi_neso);
      GeometryInterface::loc_collapsed_to_loc_coord(shape_type_int, test_eta[0], test_eta[1], test_eta[2], &xi0, &xi1, &xi2);
      ASSERT_NEAR(test_xi_neso[0], xi0, 1.0e-14);
      ASSERT_NEAR(test_xi_neso[1], xi1, 1.0e-14);
      ASSERT_NEAR(test_xi_neso[2], xi2, 1.0e-14);

      if (nektar_mapping_consistent) {
        xmap->LocCollapsedToLocCoord(test_eta, test_xi);
        ASSERT_NEAR(test_xi_neso[0], test_xi[0], 1.0e-14);
        ASSERT_NEAR(test_xi_neso[1], test_xi[1], 1.0e-14);
        ASSERT_NEAR(test_xi_neso[2], test_xi[2], 1.0e-14);
      }

      GeometryInterface::loc_coord_to_loc_collapsed_3d(shape_type_int, test_xi_neso, test_eta_neso);
      GeometryInterface::loc_coord_to_loc_collapsed_3d(shape_type_int, xi0, xi1, xi2, &eta0, &eta1, &eta2);
      ASSERT_NEAR(test_eta_neso[0], eta0, 1.0e-14);
      ASSERT_NEAR(test_eta_neso[1], eta1, 1.0e-14);
      ASSERT_NEAR(test_eta_neso[2], eta2, 1.0e-14);

      if (nektar_mapping_consistent) {
        xmap->LocCoordToLocCollapsed(test_xi_neso, test_eta);
        ASSERT_NEAR(test_eta_neso[0], test_eta[0], 1.0e-14);
        ASSERT_NEAR(test_eta_neso[1], test_eta[1], 1.0e-14);
        ASSERT_NEAR(test_eta_neso[2], test_eta[2], 1.0e-14);
      }

      vertices_found.insert(
        {
          (int) std::round(eta0),
          (int) std::round(eta1),
          (int) std::round(eta2)
        }
      );

    }
    
    // Our implementations should be consistent.
    ASSERT_EQ(vertices_found.size(), num_vertices_expected);

    /*

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

  */




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
    const int npts = xmap->GetTotPoints();
    Array<OneD, NekDouble> ptsx(npts), ptsy(npts), ptsz(npts);
    xmap->BwdTrans(geom->GetCoeffs(0), ptsx);
    xmap->BwdTrans(geom->GetCoeffs(1), ptsy);
    xmap->BwdTrans(geom->GetCoeffs(2), ptsz);
    Array<OneD, NekDouble> eta(3);
    Array<OneD, NekDouble> test_eta(3);

    xmap->LocCoordToLocCollapsed(xi, eta);
    const int shape_type_int = static_cast<int>(geom->GetShapeType());
    GeometryInterface::loc_coord_to_loc_collapsed_3d(shape_type_int, xi, test_eta);

    ASSERT_NEAR(eta[0], test_eta[0], 1.0e-8);
    ASSERT_NEAR(eta[1], test_eta[1], 1.0e-8);
    ASSERT_NEAR(eta[2], test_eta[2], 1.0e-8);

    Array<OneD, NekDouble> test_xi_neso(3);
    Array<OneD, NekDouble> test_xi_nektar(3);

    GeometryInterface::loc_collapsed_to_loc_coord(shape_type_int, eta, test_xi_neso);
    xmap->LocCollapsedToLocCoord(eta, test_xi_nektar);
    ASSERT_NEAR(test_xi_nektar[0], test_xi_neso[0], 1.0e-8);
    ASSERT_NEAR(test_xi_nektar[1], test_xi_neso[1], 1.0e-8);
    ASSERT_NEAR(test_xi_nektar[2], test_xi_neso[2], 1.0e-8);

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
    Newton::XMapNewtonTest<Newton::MappingGeneric3D> mapper(sycl_target, geom);
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

    mapper.x(test_xi0, test_xi1, test_xi2, &test_phys0, &test_phys1, &test_phys2);
    nprint("XI:", test_xi0, test_xi1, test_xi2, "PHYS:", test_phys0, test_phys1, test_phys2);

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
      EXPECT_TRUE(equiv_xi);
      EXPECT_TRUE(dist < 1.0e-8);
    }

    nprint("CHECK X MAP END");
  };

  for (auto gx : geoms) {
    auto geom = gx.second;
    nprint("GLOBALID:", geom->GetGlobalID());
    if (geom->GetGlobalID() != 45){
      continue;
    }
    auto xmap = geom->GetXmap();
    const int shape_type_int = static_cast<int>(geom->GetShapeType());
    nprint(geom->GetShapeType(), lambda_stype(geom->GetShapeType()));
    nprint("Num bases:", xmap->GetNumBases());
    for (int dx = 0; dx < 3; dx++) {
      nprint("dx:", dx, "type:", lambda_btype(xmap->GetBasisType(dx)),
             xmap->GetBasisNumModes(dx));
    }
    
    {
      NekDouble x,y,z;
      const auto num_verts = geom->GetNumVerts();
      for(int ix=0 ; ix<num_verts ; ix++){
        auto vx = geom->GetVertex(ix);
        vx->GetCoords(x,y,z);
        nprint("VX:", ix, x, y, z);
      }
    }

    Array<OneD, NekDouble> test_eta(3);
    Array<OneD, NekDouble> test_xi(3);
    Array<OneD, NekDouble> test_phys(3);

    test_eta[0] = -1.0;
    test_eta[1] = 1.0;
    test_eta[2] = -1.0;

    geom->GetXmap()->LocCollapsedToLocCoord(test_eta, test_xi);
    nprint("ETA to XI");
    nprint("\t", "ETA:", test_eta[0], test_eta[1], test_eta[2]);
    nprint("\t", " XI:", test_xi[0], test_xi[1], test_xi[2]);
    geom->GetXmap()->LocCoordToLocCollapsed(test_xi, test_eta);
    nprint("XI to ETA");
    nprint("\t", "ETA:", test_eta[0], test_eta[1], test_eta[2]);
    geom->GetXmap()->LocCollapsedToLocCoord(test_eta, test_xi);
    nprint("ETA to XI");
    nprint("\t", "ETA:", test_eta[0], test_eta[1], test_eta[2]);
    nprint("\t", " XI:", test_xi[0], test_xi[1], test_xi[2]);

    nprint("NEKTAR FORWARD");
    lambda_forward_map(geom, test_xi, test_phys);
    nprint("\t", "PHYS:", test_phys[0], test_phys[1], test_phys[2]);

    test_eta[0] = 1.0;
    test_eta[1] = -1.0;
    test_eta[2] = -1.0;

    geom->GetXmap()->LocCollapsedToLocCoord(test_eta, test_xi);
    nprint("ETA to XI");
    nprint("\t", "ETA:", test_eta[0], test_eta[1], test_eta[2]);
    nprint("\t", " XI:", test_xi[0], test_xi[1], test_xi[2]);
    geom->GetXmap()->LocCoordToLocCollapsed(test_xi, test_eta);
    nprint("XI to ETA");
    nprint("\t", "ETA:", test_eta[0], test_eta[1], test_eta[2]);
    geom->GetXmap()->LocCollapsedToLocCoord(test_eta, test_xi);
    nprint("ETA to XI");
    nprint("\t", "ETA:", test_eta[0], test_eta[1], test_eta[2]);
    nprint("\t", " XI:", test_xi[0], test_xi[1], test_xi[2]);

    nprint("NEKTAR FORWARD");
    lambda_forward_map(geom, test_xi, test_phys);
    nprint("\t", "PHYS:", test_phys[0], test_phys[1], test_phys[2]);

    {
    // Test vertices of reference element
    std::vector<std::vector<REAL>> vertices = {
      {-1.0, -1.0, -1.0},
      { 1.0, -1.0, -1.0},
      {-1.0,  1.0, -1.0},
      { 1.0,  1.0, -1.0},
      {-1.0, -1.0,  1.0},
      { 1.0, -1.0,  1.0},
      {-1.0,  1.0,  1.0},
      { 1.0,  1.0,  1.0}
    };
    
    for(auto &etav : vertices){
      nprint("==============================================");
      test_eta[0] = etav.at(0);
      test_eta[1] = etav.at(1);
      test_eta[2] = etav.at(2);
      
      // GeometryInterface::loc_collapsed_to_loc_coord(shape_type_int, test_eta, test_xi);

      nprint("ETA 0:", test_eta[0], test_eta[1], test_eta[2]);
      geom->GetXmap()->LocCollapsedToLocCoord(test_eta, test_xi);
      // GeometryInterface::loc_collapsed_to_loc_coord(shape_type_int, test_eta, test_xi);
      nprint(" XI 0:", test_xi[0], test_xi[1], test_xi[2]);
      geom->GetXmap()->LocCoordToLocCollapsed(test_xi, test_eta);
      nprint("ETA 1:", test_eta[0], test_eta[1], test_eta[2]);
      geom->GetXmap()->LocCollapsedToLocCoord(test_eta, test_xi);
      // GeometryInterface::loc_collapsed_to_loc_coord(shape_type_int, test_eta, test_xi);
      nprint(" XI 1:", test_xi[0], test_xi[1], test_xi[2]);
      geom->GetXmap()->LocCoordToLocCollapsed(test_xi, test_eta);
      nprint("ETA 2:", test_eta[0], test_eta[1], test_eta[2]);


      lambda_forward_map(geom, test_xi, test_phys);
      nprint("TEST: XI:", test_xi[0], test_xi[1], test_xi[2], "\n   ETA O:", test_eta[0], test_eta[1], test_eta[2], "\nPHYS:", test_phys[0], test_phys[1], test_phys[2]);

      // Does nektar sucessfully invert the map?
      Array<OneD, NekDouble> test_xi_nektar(3);
      auto dist = geom->GetLocCoords(test_phys, test_xi_nektar);
      nprint("TEST: NK:", test_xi_nektar[0], test_xi_nektar[1], test_xi_nektar[2], "DIST:", dist);
    }

    nprint("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");

    for(auto &etav : vertices){
      nprint("==============================================");
      test_xi[0] = etav.at(0);
      test_xi[1] = etav.at(1);
      test_xi[2] = etav.at(2);

      if ((test_xi[0] + test_xi[1] + test_xi[2]) <= -0.9){
      geom->GetXmap()->LocCoordToLocCollapsed(test_xi, test_eta);
      nprint("ETA::", test_eta[0], test_eta[1], test_eta[2]);

      lambda_forward_map(geom, test_xi, test_phys);
      nprint("TEST: XI:", test_xi[0], test_xi[1], test_xi[2], "PHYS:", test_phys[0], test_phys[1], test_phys[2]);

      // Does nektar sucessfully invert the map?
      Array<OneD, NekDouble> test_xi_nektar(3);
      auto dist = geom->GetLocCoords(test_phys, test_xi_nektar);
      nprint("TEST: NK:", test_xi_nektar[0], test_xi_nektar[1], test_xi_nektar[2], "DIST:", dist);
      }
    }




    }



    // Test vertices of reference element
    std::vector<std::vector<REAL>> vertices = {
      //{-1.0, -1.0, -1.0},
      { 1.0, -1.0, -1.0},
      //{-1.0,  1.0, -1.0},
      //{ 1.0,  1.0, -1.0},
      //{-1.0, -1.0,  1.0},
      //{ 1.0, -1.0,  1.0},
      //{-1.0,  1.0,  1.0},
      //{ 1.0,  1.0,  1.0}
    };
    
    for(auto &etav : vertices){
      nprint("----------------------------------------------");
      test_eta[0] = etav.at(0);
      test_eta[1] = etav.at(1);
      test_eta[2] = etav.at(2);
      GeometryInterface::loc_collapsed_to_loc_coord(shape_type_int, test_eta, test_xi);
      lambda_forward_map(geom, test_xi, test_phys);
      nprint("TEST: XI:", test_xi[0], test_xi[1], test_xi[2], "\nETA:", test_eta[0], test_eta[1], test_eta[2], "\nPHYS:", test_phys[0], test_phys[1], test_phys[2]);


      // Does nektar sucessfully invert the map?
      Array<OneD, NekDouble> test_xi_nektar(3);
      auto dist = geom->GetLocCoords(test_phys, test_xi_nektar);
      nprint("TEST: NK:", test_xi_nektar[0], test_xi_nektar[1], test_xi_nektar[2], "DIST:", dist);

      lambda_check_x_map(geom, test_xi, test_phys);
    }

    // test internal points
    //for(int testx=0 ; testx<20 ; testx++){
    //  nprint("==============================================");
    //  lambda_sample_internal_point(geom, test_xi, test_phys);
    //  lambda_check_x_map(geom, test_xi, test_phys);
    //}
  }

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
