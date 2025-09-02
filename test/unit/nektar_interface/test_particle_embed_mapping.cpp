#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <gtest/gtest.h>

#include "nektar_interface/composite_interaction/composite_interaction.hpp"
#include "nektar_interface/particle_cell_mapping/newton_geom_interfaces.hpp"
#include "nektar_interface/particle_interface.hpp"

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;
using namespace NESO::CompositeInteraction;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

template <typename U, typename R>
static inline void get_point_in_and_out_plane(U &geom, R &rng, REAL *point_in,
                                              REAL *point_out) {

  std::uniform_real_distribution<double> ref_distribution(-1.0, 1.0);
  Array<OneD, NekDouble> xi(3);
  Array<OneD, NekDouble> cg(3);
  REAL g[3];

  // Get a point in the reference element
  cg[0] = ref_distribution(rng);
  cg[1] = ref_distribution(rng);
  cg[2] = 0.0;

  xi[0] = 0.0;
  xi[1] = 0.0;
  xi[2] = 0.0;
  geom->GetXmap()->LocCollapsedToLocCoord(cg, xi);

  // check the map from reference space to global space
  for (int dx = 0; dx < 3; dx++) {
    point_in[dx] = geom->GetCoord(dx, xi);
    point_out[dx] = -1.0 * point_in[dx];
  }
}

template <typename T, typename U, typename R>
static inline void check_geom_map(T &n, U &geom, R &rng) {

  const int N_test = 5;
  std::uniform_real_distribution<double> ref_distribution(-1.0, 1.0);
  Array<OneD, NekDouble> xi(3);
  Array<OneD, NekDouble> cg(3);
  REAL g[3];

  for (int testx = 0; testx < N_test; testx++) {

    // Get a point in the reference element
    cg[0] = ref_distribution(rng);
    cg[1] = ref_distribution(rng);
    cg[2] = 0.0;

    xi[0] = 0.0;
    xi[1] = 0.0;
    xi[2] = 0.0;
    geom->GetXmap()->LocCollapsedToLocCoord(cg, xi);

    const int ndim = geom->GetCoordim();
    n.x(xi[0], xi[1], xi[2], g, g + 1, g + 2);
    // check the map from reference space to global space
    for (int dx = 0; dx < ndim; dx++) {
      cg[dx] = geom->GetCoord(dx, xi);
      if (std::isfinite(cg[dx])) {
        const REAL err_abs = abs(cg[dx] - g[dx]);
        const REAL err = std::min(err_abs, err_abs / abs(cg[dx]));
        ASSERT_TRUE(err < 1.0e-12);
      }
    }

    // check the map from global space back to reference space
    REAL xi_check[3];
    n.x_inverse(g[0], g[1], g[2], xi_check, xi_check + 1, xi_check + 2);
    for (int dx = 0; dx < ndim; dx++) {
      const REAL err_abs = abs(xi_check[dx] - xi[dx]);
      const REAL err_rel_t = err_abs / std::abs(xi[dx]);
      const REAL err_rel = std::isnormal(err_rel_t) ? err_rel_t : err_abs;
      const REAL err = std::min(err_abs, err_rel);
      // The exit tol on the newton method is 1E-10 so we test against 1E-8.
      ASSERT_TRUE(err < 1.0e-8);
    }
  }
}

TEST(EmbeddedXMapping, Base) {
  const int N_total = 2000;

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
  graph = SpatialDomains::MeshGraphIO::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto graph_composites = graph->GetComposites();

  std::vector<int> compsite_indices(6);
  compsite_indices[0] = 100;
  compsite_indices[1] = 200;
  compsite_indices[2] = 300;
  compsite_indices[3] = 400;
  compsite_indices[4] = 500;
  compsite_indices[5] = 600;
  std::mt19937 rng(182348 + sycl_target->comm_pair.rank_parent);

  for (auto cx : compsite_indices) {
    if (graph_composites.count(cx)) {
      auto geoms = graph_composites.at(cx)->m_geomVec;
      for (auto &geom : geoms) {
        if (geom->GetShapeType() == eQuadrilateral) {
          auto n = Newton::XMapNewton<Newton::MappingQuadLinear2DEmbed3D>(
              sycl_target, geom);
          check_geom_map(n, geom, rng);
        } else if (geom->GetShapeType() == eTriangle) {
          auto n = Newton::XMapNewton<Newton::MappingTriangleLinear2DEmbed3D>(
              sycl_target, geom);
          check_geom_map(n, geom, rng);
        }
      }
    }
  }

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
}

TEST(EmbeddedXMapping, LinePlaneIntersection) {
  const int N_total = 2000;

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
  graph = SpatialDomains::MeshGraphIO::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto graph_composites = graph->GetComposites();

  std::vector<int> compsite_indices(6);
  compsite_indices[0] = 100;
  compsite_indices[1] = 200;
  compsite_indices[2] = 300;
  compsite_indices[3] = 400;
  compsite_indices[4] = 500;
  compsite_indices[5] = 600;
  std::mt19937 rng(182348 + sycl_target->comm_pair.rank_parent);

  for (auto cx : compsite_indices) {
    if (graph_composites.count(cx)) {
      auto geoms = graph_composites.at(cx)->m_geomVec;
      for (auto &geom : geoms) {

        LinePlaneIntersection line_plane_intersection(geom);
        REAL point_in[3], point_out[3], point_int[3], origin[3];
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        get_point_in_and_out_plane(geom, rng, point_in, point_out);
        bool exists;
        // line stops in the plane
        exists = line_plane_intersection.line_segment_intersection(
            origin[0], origin[1], origin[2], point_in[0], point_in[1],
            point_in[2], point_int, point_int + 1, point_int + 2);
        ASSERT_TRUE(exists);
        ASSERT_NEAR(point_in[0], point_int[0], 1.0e-13);
        ASSERT_NEAR(point_in[1], point_int[1], 1.0e-13);
        ASSERT_NEAR(point_in[2], point_int[2], 1.0e-13);
        // line points away from the plane
        exists = line_plane_intersection.line_segment_intersection(
            origin[0], origin[1], origin[2], point_out[0], point_out[1],
            point_out[2], point_int, point_int + 1, point_int + 2);
        ASSERT_TRUE(!exists);
        // trivial line in the plane
        exists = line_plane_intersection.line_segment_intersection(
            point_in[0], point_in[1], point_in[2], point_in[0], point_in[1],
            point_in[2], point_int, point_int + 1, point_int + 2);
        ASSERT_TRUE(exists);
        ASSERT_NEAR(point_in[0], point_int[0], 1.0e-13);
        ASSERT_NEAR(point_in[1], point_int[1], 1.0e-13);
        ASSERT_NEAR(point_in[2], point_int[2], 1.0e-13);
        // trivial line out of the plane
        exists = line_plane_intersection.line_segment_intersection(
            origin[0], origin[1], origin[2], origin[0], origin[1], origin[2],
            point_int, point_int + 1, point_int + 2);
        ASSERT_TRUE(!exists);
        // line stops short of plane
        REAL point_short[3];
        point_short[0] = point_in[0] * 0.95;
        point_short[1] = point_in[1] * 0.95;
        point_short[2] = point_in[2] * 0.95;
        exists = line_plane_intersection.line_segment_intersection(
            origin[0], origin[1], origin[2], point_short[0], point_short[1],
            point_short[2], point_int, point_int + 1, point_int + 2);
        ASSERT_TRUE(!exists);
        // line goes through plane
        REAL point_long[3];
        point_long[0] = point_in[0] * 1.5;
        point_long[1] = point_in[1] * 1.5;
        point_long[2] = point_in[2] * 1.5;
        exists = line_plane_intersection.line_segment_intersection(
            origin[0], origin[1], origin[2], point_long[0], point_long[1],
            point_long[2], point_int, point_int + 1, point_int + 2);
        ASSERT_TRUE(exists);
        ASSERT_NEAR(point_in[0], point_int[0], 1.0e-13);
        ASSERT_NEAR(point_in[1], point_int[1], 1.0e-13);
        ASSERT_NEAR(point_in[2], point_int[2], 1.0e-13);
        // non trivial line in the plane
        NekDouble x, y, z;
        geom->GetVertex(0)->GetCoords(x, y, z);
        REAL point_in2[3] = {x, y, z};
        exists = line_plane_intersection.line_segment_intersection(
            point_in2[0], point_in2[1], point_in2[2], point_in[0], point_in[1],
            point_in[2], point_int, point_int + 1, point_int + 2);
        ASSERT_TRUE(exists);
        ASSERT_NEAR(point_in2[0], point_int[0], 1.0e-13);
        ASSERT_NEAR(point_in2[1], point_int[1], 1.0e-13);
        ASSERT_NEAR(point_in2[2], point_int[2], 1.0e-13);
        // line starts in the plane
        exists = line_plane_intersection.line_segment_intersection(
            point_in[0], point_in[1], point_in[2], point_out[0], point_out[1],
            point_out[2], point_int, point_int + 1, point_int + 2);
        ASSERT_TRUE(exists);
        ASSERT_NEAR(point_in[0], point_int[0], 1.0e-13);
        ASSERT_NEAR(point_in[1], point_int[1], 1.0e-13);
        ASSERT_NEAR(point_in[2], point_int[2], 1.0e-13);

        // test the is near functionality
        const int num_verts = geom->GetNumVerts();

        for (int vx = 0; vx < num_verts; vx++) {

          NekDouble t0, t1, t2;
          auto vertex = geom->GetVertex(vx);
          vertex->GetCoords(t0, t1, t2);
          ASSERT_TRUE(line_plane_intersection.point_near_to_geom(t0, t1, t2));
        }

        ASSERT_TRUE(line_plane_intersection.point_near_to_geom(
            point_in[0], point_in[1], point_in[2]));
      }
    }
  }

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
}

TEST(EmbeddedXMapping, LineLineIntersection) {

  auto lambda_make_seggeom = [&](auto a, auto b) {
    std::vector<SpatialDomains::PointGeomSharedPtr> points = {
        std::make_shared<SpatialDomains::PointGeom>(2, 0, a[0], a[1], 0.0),
        std::make_shared<SpatialDomains::PointGeom>(2, 1, b[0], b[1], 0.0)};
    return std::make_shared<SpatialDomains::SegGeom>(0, 2, points.data());
  };

  auto lambda_test_line = [&](auto lli, auto x, auto y, auto *i) -> bool {
    return lli.line_line_intersection(x[0], x[1], y[0], y[1], i, i + 1);
  };

  auto lambda_test_ref_coord = [&](auto lli, auto i) -> REAL {
    return lli.get_reference_coordinate(i[0], i[1]);
  };

  REAL i[2] = {0.0, 0.0};

  {
    REAL a[2] = {0.0, 0.0};
    REAL b[2] = {1.0, 0.0};
    auto s = lambda_make_seggeom(a, b);
    auto lli = CompositeInteraction::LineLineIntersection(s);
    REAL x[2] = {0.5, -0.5};
    REAL y[2] = {0.5, 0.5};
    ASSERT_TRUE(lambda_test_line(lli, x, y, i));
    ASSERT_NEAR(i[0], 0.5, 1.0e-14);
    ASSERT_NEAR(i[1], 0.0, 1.0e-14);

    ASSERT_NEAR(lli.normalx, 0.0, 1.0e-14);
    ASSERT_NEAR(lli.normaly, 1.0, 1.0e-14);

    REAL xi = lambda_test_ref_coord(lli, i);
    ASSERT_NEAR(xi, 0.0, 1.0e-14);
    xi = lambda_test_ref_coord(lli, a);
    ASSERT_NEAR(xi, -1.0, 1.0e-14);
    xi = lambda_test_ref_coord(lli, b);
    ASSERT_NEAR(xi, 1.0, 1.0e-14);
    REAL t[2] = {0.25, 0.0};
    xi = lambda_test_ref_coord(lli, t);
    ASSERT_NEAR(xi, 0.25 * 2.0 - 1.0, 1.0e-14);
  }
  {
    REAL a[2] = {0.5, -0.5};
    REAL b[2] = {0.5, 0.5};
    auto s = lambda_make_seggeom(a, b);
    auto lli = CompositeInteraction::LineLineIntersection(s);
    REAL x[2] = {0.0, 0.0};
    REAL y[2] = {1.0, 0.0};
    ASSERT_TRUE(lambda_test_line(lli, x, y, i));
    ASSERT_NEAR(i[0], 0.5, 1.0e-14);
    ASSERT_NEAR(i[1], 0.0, 1.0e-14);

    ASSERT_NEAR(lli.normalx, -1.0, 1.0e-14);
    ASSERT_NEAR(lli.normaly, 0.0, 1.0e-14);
  }

  {
    REAL a[2] = {0.5, -0.5};
    REAL b[2] = {0.5, -0.0001};
    auto s = lambda_make_seggeom(a, b);
    auto lli = CompositeInteraction::LineLineIntersection(s);
    REAL x[2] = {0.0, 0.0};
    REAL y[2] = {1.0, 0.0};
    ASSERT_FALSE(lambda_test_line(lli, x, y, i));
  }
}
