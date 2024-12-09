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
  auto graph = SpatialDomains::MeshGraphIO::Read(session);

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

    auto bb = BoundingBox::get_bounding_box(sycl_target, h, 32, 0.05, 0.0);
  }

  sycl_target->free();
}
