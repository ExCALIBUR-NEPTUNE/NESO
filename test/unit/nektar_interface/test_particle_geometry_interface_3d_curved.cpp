#include "test_helper_utilities.hpp"

TEST(ParticleGeometryInterfaceCurved, XMapNewtonBase) {

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  TestUtilities::TestResourceSession resource_session(
      "reference_all_types_cube/mixed_ref_cube_0.5_perturbed_order_2.xml",
      "reference_all_types_cube/conditions.xml");

  // Create session reader.
  auto session = resource_session.session;

  // Create MeshGraph.
  auto graph = SpatialDomains::MeshGraph::Read(session);

  // build map from owned mesh hierarchy cells to geoms that touch that cell
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms;
  get_all_elements_3d(graph, geoms);

  auto lambda_forward_map = [&](auto geom, const auto &xi, auto &phys) {
    // Evaluate the forward map from xi to physical space using the expansion.
    auto xmap = geom->GetXmap();
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
      EXPECT_TRUE(equiv_xi);
      EXPECT_TRUE(dist < 1.0e-8);
    }
  };

  for (auto gx : geoms) {
    auto geom = gx.second;
    const int shape_type_int = static_cast<int>(geom->GetShapeType());

    Array<OneD, NekDouble> test_eta(3);
    Array<OneD, NekDouble> test_xi(3);
    Array<OneD, NekDouble> test_phys(3);

    // Test vertices of reference element
    std::vector<std::vector<REAL>> vertices = {
        {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0},
        {1.0, 1.0, -1.0},   {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
        {-1.0, 1.0, 1.0},   {1.0, 1.0, 1.0}};

    for (auto &etav : vertices) {
      test_eta[0] = etav.at(0);
      test_eta[1] = etav.at(1);
      test_eta[2] = etav.at(2);
      GeometryInterface::loc_collapsed_to_loc_coord(shape_type_int, test_eta,
                                                    test_xi);
      lambda_forward_map(geom, test_xi, test_phys);
      lambda_check_x_map(geom, test_xi, test_phys);
    }

    // test internal points
    for (int testx = 0; testx < 5; testx++) {
      lambda_sample_internal_point(geom, test_xi, test_phys);
      lambda_check_x_map(geom, test_xi, test_phys);
    }
  }

  sycl_target->free();
  mesh->free();
}

template <typename T, typename U, typename V>
inline std::shared_ptr<HexGeom> make_hex_geom(const int num_modes, T xmapx,
                                              U xmapy, V xmapz) {

  /**
   * Vertices:
   *
   *  7 - 6
   *  |   |
   *  4 - 5
   *
   *  3 - 2
   *  |   |
   *  0 - 1
   *
   *  Edges:
   *
   *  * 10 *
   *  11   9
   *  *  8 *
   *
   *  7 - 6
   *  |   |
   *  4 - 5
   *
   *  * 2 *
   *  3   1
   *  * 0 *
   *
   *  Faces:
   *
   *  * - *
   *  | 5 |
   *  * - *
   *
   *  * 3 *
   *  4   2
   *  * 1 *
   *
   *  * - *
   *  | 0 |
   *  * - *
   *
   *  auto pts = LibUtilities::PointsManager()[*points_key];
   *  Triangle is something like trievenlyspaced
   *  quad will x fastest then y
   *  Triangles are not the expansion looping ordering - look at nektmesh
   *  top eta0, eta1, eta2=0
   *  eta0 eta1
   *
   *  make meshgraph, meshgraphio. view -> adavnced "nonlinear subdivisions
   *  (slider)"
   */

  REAL coords_vertices[8][3] = {{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0},
                                {1.0, 1.0, -1.0},   {-1.0, 1.0, -1.0},
                                {-1.0, -1.0, 1.0},  {1.0, -1.0, 1.0},
                                {1.0, 1.0, 1.0},    {-1.0, 1.0, 1.0}};

  int map_edge_to_vertices[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},

      {0, 4}, {1, 5}, {2, 6}, {3, 7},

      {4, 5}, {5, 6}, {6, 7}, {7, 4},
  };

  int map_face_to_edges[6][4] = {
      {0, 1, 2, 3},  {0, 5, 8, 4},  {1, 6, 9, 5},
      {2, 7, 10, 6}, {3, 4, 11, 7}, {8, 9, 10, 11},
  };

  int map_face_to_vertices[6][4] = {{0, 1, 2, 3}, {0, 1, 4, 5}, {1, 2, 6, 5},
                                    {2, 3, 7, 6}, {1, 3, 7, 4}, {4, 5, 6, 7}};

  auto points_key = PointsKey(num_modes, ePolyEvenlySpaced);
  auto control_points = std::make_shared<PolyEPoints>(points_key);
  control_points->Initialize();
  auto Z = control_points->GetZ();
  std::vector<REAL> Znormalised(Z.size());
  for (int ix = 0; ix < num_modes; ix++) {
    Znormalised.at(ix) = (Z[ix] + 1.0) / 2.0;
  }

  std::map<int, std::shared_ptr<PointGeom>> v;
  std::map<int, std::shared_ptr<SegGeom>> e;
  std::map<int, std::shared_ptr<Curve>> c;
  std::map<int, std::shared_ptr<QuadGeom>> q;

  // Create the vertices
  int vx_index = 0;
  int cx_index = 0;
  for (int vx = 0; vx < 8; vx++) {
    const REAL px = xmapx(coords_vertices[vx]);
    const REAL py = xmapy(coords_vertices[vx]);
    const REAL pz = xmapz(coords_vertices[vx]);
    v[vx] = std::make_shared<PointGeom>(3, vx_index++, px, py, pz);
  }

  // Create the edges
  auto lambda_get_1D_qpoint = [&](const int point_id, const auto a,
                                  const auto b) -> std::array<REAL, 3> {
    NekDouble ax, ay, az, bx, by, bz;
    ax = a[0];
    ay = a[1];
    az = a[2];
    bx = b[0];
    by = b[1];
    bz = b[2];
    const NekDouble dx = bx - ax;
    const NekDouble dy = by - ay;
    const NekDouble dz = bz - az;

    std::array<REAL, 3> out;
    out[0] = ax + Znormalised[point_id] * dx;
    out[1] = ay + Znormalised[point_id] * dy;
    out[2] = az + Znormalised[point_id] * dz;
    return out;
  };

  for (int ex = 0; ex < 12; ex++) {
    auto cx = std::make_shared<Curve>(cx_index++, ePolyEvenlySpaced);

    for (int mx = 0; mx < num_modes; mx++) {
      auto ref_coord =
          lambda_get_1D_qpoint(mx, coords_vertices[map_edge_to_vertices[ex][0]],
                               coords_vertices[map_edge_to_vertices[ex][1]]);
      const REAL px = xmapx(ref_coord);
      const REAL py = xmapy(ref_coord);
      const REAL pz = xmapz(ref_coord);
      cx->m_points.push_back(
          std::make_shared<PointGeom>(3, vx_index++, px, py, pz));
    }
    std::shared_ptr<PointGeom> vertices_array[2] = {
        v.at(map_edge_to_vertices[ex][0]), v.at(map_edge_to_vertices[ex][1])};
    c[ex] = cx;
    e[ex] = std::make_shared<SegGeom>(ex, 3, vertices_array, cx);
    e[ex]->GetGeomFactors();
    e[ex]->Setup();
  }

  // Create the faces
  auto lambda_get_2D_qpoint = [&](const int point_id0, const int point_id1,
                                  const auto a, const auto b,
                                  const auto c) -> std::array<REAL, 3> {
    /**
     * c
     * |
     * a - b
     */

    NekDouble ax, ay, az, bx, by, bz, cx, cy, cz;
    ax = a[0];
    ay = a[1];
    az = a[2];
    bx = b[0];
    by = b[1];
    bz = b[2];
    cx = c[0];
    cy = c[1];
    cz = c[2];
    const NekDouble d0x = bx - ax;
    const NekDouble d0y = by - ay;
    const NekDouble d0z = bz - az;
    const NekDouble d1x = cx - ax;
    const NekDouble d1y = cy - ay;
    const NekDouble d1z = cz - az;

    std::array<REAL, 3> out;
    out[0] = ax + Znormalised[point_id0] * d0x + Znormalised[point_id1] * d1x;
    out[1] = ay + Znormalised[point_id0] * d0y + Znormalised[point_id1] * d1y;
    out[2] = az + Znormalised[point_id0] * d0z + Znormalised[point_id1] * d1z;
    return out;
  };

  for (int fx = 0; fx < 6; fx++) {
    std::shared_ptr<PointGeom> vertices_array[4] = {
        v.at(map_face_to_vertices[fx][0]), v.at(map_face_to_vertices[fx][1]),
        v.at(map_face_to_vertices[fx][2]), v.at(map_face_to_vertices[fx][3])};

    auto cx = std::make_shared<Curve>(cx_index++, ePolyEvenlySpaced);
    for (int mx = 0; mx < num_modes; mx++) {
      for (int my = 0; my < num_modes; my++) {
        auto ref_coord = lambda_get_2D_qpoint(
            mx, my, coords_vertices[map_face_to_vertices[fx][0]],
            coords_vertices[map_face_to_vertices[fx][1]],
            coords_vertices[map_face_to_vertices[fx][3]]);
        const REAL px = xmapx(ref_coord);
        const REAL py = xmapy(ref_coord);
        const REAL pz = xmapz(ref_coord);
        cx->m_points.push_back(
            std::make_shared<PointGeom>(3, vx_index++, px, py, pz));
      }
    }

    std::shared_ptr<SegGeom> edges_array[4] = {
        e.at(map_face_to_edges[fx][0]), e.at(map_face_to_edges[fx][1]),
        e.at(map_face_to_edges[fx][2]), e.at(map_face_to_edges[fx][3])};
    q[fx] = std::make_shared<QuadGeom>(fx, edges_array, cx);
    q[fx]->GetGeomFactors();
    q[fx]->Setup();
  }

  std::shared_ptr<QuadGeom> quads[6] = {q[0], q[1], q[2], q[3], q[4], q[5]};

  auto hex = std::make_shared<HexGeom>(0, quads);
  hex->GetGeomFactors();
  hex->Setup();

  return hex;
}

TEST(ParticleGeometryInterfaceCurved, MakeCurvedHex) {

  auto lambda_check = [&](auto correct, auto to_test) {
    const auto err_abs = std::abs(correct - to_test);
    const auto err_rel =
        std::abs(correct) > 0.0 ? err_abs / std::abs(correct) : err_abs;
    const REAL tol = 1.0e-10;
    ASSERT_TRUE(err_abs < tol || err_rel < tol);
  };

  {
    auto xmapx = [&](auto eta) { return eta[0] * 2.0; };
    auto xmapy = [&](auto eta) { return eta[1] * 3.0; };
    auto xmapz = [&](auto eta) { return eta[2] * 5.0; };

    const int num_modes = 3;
    auto h = make_hex_geom(num_modes, xmapx, xmapy, xmapz);

    auto points_key = PointsKey(num_modes, ePolyEvenlySpaced);
    auto control_points = std::make_shared<PolyEPoints>(points_key);
    control_points->Initialize();
    auto Z = control_points->GetZ();

    Array<OneD, NekDouble> lcoords(3);
    for (int iz = 0; iz < num_modes; iz++) {
      for (int iy = 0; iy < num_modes; iy++) {
        for (int ix = 0; ix < num_modes; ix++) {
          lcoords[0] = Z[ix];
          lcoords[1] = Z[iy];
          lcoords[2] = Z[iz];

          auto x0 = h->GetCoord(0, lcoords);
          auto x1 = h->GetCoord(1, lcoords);
          auto x2 = h->GetCoord(2, lcoords);
          auto c0 = xmapx(lcoords);
          auto c1 = xmapy(lcoords);
          auto c2 = xmapz(lcoords);
          lambda_check(c0, x0);
          lambda_check(c1, x1);
          lambda_check(c2, x2);
        }
      }
    }
  }

  {
    auto xmapx = [&](auto eta) { return eta[0] * 2.0 + 7.0; };
    auto xmapy = [&](auto eta) { return eta[1] * 3.0 + 9.0; };
    auto xmapz = [&](auto eta) { return eta[2] * 5.0 + 11.0; };

    const int num_modes = 3;
    auto h = make_hex_geom(num_modes, xmapx, xmapy, xmapz);

    auto points_key = PointsKey(num_modes, ePolyEvenlySpaced);
    auto control_points = std::make_shared<PolyEPoints>(points_key);
    control_points->Initialize();
    auto Z = control_points->GetZ();

    Array<OneD, NekDouble> lcoords(3);
    for (int iz = 0; iz < num_modes; iz++) {
      for (int iy = 0; iy < num_modes; iy++) {
        for (int ix = 0; ix < num_modes; ix++) {
          lcoords[0] = Z[ix];
          lcoords[1] = Z[iy];
          lcoords[2] = Z[iz];

          auto x0 = h->GetCoord(0, lcoords);
          auto x1 = h->GetCoord(1, lcoords);
          auto x2 = h->GetCoord(2, lcoords);
          auto c0 = xmapx(lcoords);
          auto c1 = xmapy(lcoords);
          auto c2 = xmapz(lcoords);
          lambda_check(c0, x0);
          lambda_check(c1, x1);
          lambda_check(c2, x2);
        }
      }
    }
  }

  {
    auto xmapx = [&](auto eta) { return eta[0]; };
    auto xmapy = [&](auto eta) { return eta[1]; };
    auto xmapz = [&](auto eta) {
      return eta[2] + 0.2 * eta[0] * eta[0] + 0.2 * eta[1] * eta[1];
    };

    const int num_modes = 3;
    auto h = make_hex_geom(num_modes, xmapx, xmapy, xmapz);

    auto points_key = PointsKey(num_modes, ePolyEvenlySpaced);
    auto control_points = std::make_shared<PolyEPoints>(points_key);
    control_points->Initialize();
    auto Z = control_points->GetZ();

    Array<OneD, NekDouble> lcoords(3);
    for (int iz = 0; iz < num_modes; iz++) {
      for (int iy = 0; iy < num_modes; iy++) {
        for (int ix = 0; ix < num_modes; ix++) {
          lcoords[0] = Z[ix];
          lcoords[1] = Z[iy];
          lcoords[2] = Z[iz];

          auto x0 = h->GetCoord(0, lcoords);
          auto x1 = h->GetCoord(1, lcoords);
          auto x2 = h->GetCoord(2, lcoords);
          auto c0 = xmapx(lcoords);
          auto c1 = xmapy(lcoords);
          auto c2 = xmapz(lcoords);
          lambda_check(c0, x0);
          lambda_check(c1, x1);
          lambda_check(c2, x2);
        }
      }
    }
  }

  {
    auto xmapx = [&](auto eta) { return eta[0]; };
    auto xmapy = [&](auto eta) {
      return eta[1] + 0.1 * eta[0] + 0.1 * eta[2] * eta[2];
    };
    auto xmapz = [&](auto eta) {
      return eta[2] + 0.2 * eta[0] * eta[0] + 0.2 * eta[1] * eta[1];
    };

    const int num_modes = 3;
    auto h = make_hex_geom(num_modes, xmapx, xmapy, xmapz);

    auto points_key = PointsKey(num_modes, ePolyEvenlySpaced);
    auto control_points = std::make_shared<PolyEPoints>(points_key);
    control_points->Initialize();
    auto Z = control_points->GetZ();

    Array<OneD, NekDouble> lcoords(3);
    for (int iz = 0; iz < num_modes; iz++) {
      for (int iy = 0; iy < num_modes; iy++) {
        for (int ix = 0; ix < num_modes; ix++) {
          lcoords[0] = Z[ix];
          lcoords[1] = Z[iy];
          lcoords[2] = Z[iz];

          auto x0 = h->GetCoord(0, lcoords);
          auto x1 = h->GetCoord(1, lcoords);
          auto x2 = h->GetCoord(2, lcoords);
          auto c0 = xmapx(lcoords);
          auto c1 = xmapy(lcoords);
          auto c2 = xmapz(lcoords);
          lambda_check(c0, x0);
          lambda_check(c1, x1);
          lambda_check(c2, x2);
        }
      }
    }
  }

  {
    auto xmapx = [&](auto eta) {
      return eta[0] + (0.2 - 0.1 * eta[2] * eta[2]) - 2.0;
    };
    auto xmapy = [&](auto eta) {
      return eta[1] + 0.1 * eta[0] + 0.1 * eta[2] * eta[2];
    };
    auto xmapz = [&](auto eta) {
      return eta[2] + 0.2 * eta[0] * eta[0] + 0.2 * eta[1] * eta[1];
    };

    const int num_modes = 3;
    auto h = make_hex_geom(num_modes, xmapx, xmapy, xmapz);

    auto points_key = PointsKey(num_modes, ePolyEvenlySpaced);
    auto control_points = std::make_shared<PolyEPoints>(points_key);
    control_points->Initialize();
    auto Z = control_points->GetZ();

    Array<OneD, NekDouble> lcoords(3);
    for (int iz = 0; iz < num_modes; iz++) {
      for (int iy = 0; iy < num_modes; iy++) {
        for (int ix = 0; ix < num_modes; ix++) {
          lcoords[0] = Z[ix];
          lcoords[1] = Z[iy];
          lcoords[2] = Z[iz];

          auto x0 = h->GetCoord(0, lcoords);
          auto x1 = h->GetCoord(1, lcoords);
          auto x2 = h->GetCoord(2, lcoords);
          auto c0 = xmapx(lcoords);
          auto c1 = xmapy(lcoords);
          auto c2 = xmapz(lcoords);
          lambda_check(c0, x0);
          lambda_check(c1, x1);
          lambda_check(c2, x2);
        }
      }
    }
  }
}

TEST(ParticleGeometryInterfaceCurved, BoundingBox) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  sycl_target->free();
}
