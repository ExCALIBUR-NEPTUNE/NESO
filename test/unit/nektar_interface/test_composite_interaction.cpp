#include <SpatialDomains/MeshGraphIO.h>

#include "../../unit/nektar_interface/test_helper_utilities.hpp"

using namespace CompositeInteraction;

namespace {

class CompositeIntersectionTester
    : public CompositeInteraction::CompositeIntersection {

public:
  inline void test_find_cells(ParticleGroupSharedPtr particle_group,
                              std::set<INT> &cells) {
    return this->find_cells(particle_group, cells);
  }

  inline std::shared_ptr<CompositeCollections> &get_composite_collections() {
    return this->composite_collections;
  }

  CompositeIntersectionTester(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      std::map<int, std::vector<int>> &boundary_groups)
      : CompositeIntersection(sycl_target, particle_mesh_interface,
                              boundary_groups) {}
};

class CompositeTransportTester
    : public CompositeInteraction::CompositeTransport {

public:
  CompositeTransportTester(
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      std::vector<int> &composite_indices)
      : CompositeTransport(particle_mesh_interface, composite_indices) {}

  inline auto &get_contrib_cells() { return this->contrib_cells; }
};

} // namespace

TEST(CompositeInteraction, AtomicFetchMaxMin) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  typedef int TEST_INT;

  std::vector<TEST_INT> h_buffer = {0, 0};
  auto dh_buffer = BufferDeviceHost<TEST_INT>(sycl_target, h_buffer);
  auto k_buffer = dh_buffer.d_buffer.ptr;

  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(1), [=](sycl::id<1> idx) {
          atomic_fetch_max(k_buffer, (TEST_INT)8);
          atomic_fetch_min(&k_buffer[1], (TEST_INT)-8);
        });
      })
      .wait_and_throw();

  dh_buffer.device_to_host();

  EXPECT_EQ(dh_buffer.h_buffer.ptr[0], 8);
  EXPECT_EQ(dh_buffer.h_buffer.ptr[1], -8);

  sycl_target->free();
}

TEST(CompositeInteraction, Utility) {

  auto lambda_make_edge = [&](auto a, auto b) {
    std::vector<SpatialDomains::PointGeomSharedPtr> vertices;
    vertices.push_back(
        std::make_shared<SpatialDomains::PointGeom>(3, 0, a[0], a[1], a[2]));
    vertices.push_back(
        std::make_shared<SpatialDomains::PointGeom>(3, 1, b[0], b[1], b[2]));
    return std::dynamic_pointer_cast<SpatialDomains::Geometry1D>(
        std::make_shared<SpatialDomains::SegGeom>(0, 3, vertices.data()));
  };

  auto lambda_make_triangle = [&](auto a, auto b, auto c) {
    std::vector<SpatialDomains::SegGeomSharedPtr> edges = {
        std::dynamic_pointer_cast<SpatialDomains::SegGeom>(
            lambda_make_edge(a, b)),
        std::dynamic_pointer_cast<SpatialDomains::SegGeom>(
            lambda_make_edge(b, c)),
        std::dynamic_pointer_cast<SpatialDomains::SegGeom>(
            lambda_make_edge(c, a))};
    return std::dynamic_pointer_cast<Geometry2D>(
        std::make_shared<SpatialDomains::TriGeom>(0, edges.data()));
  };

  auto lambda_make_quad = [&](auto a, auto b, auto c, auto d) {
    std::vector<SpatialDomains::SegGeomSharedPtr> edges = {
        std::dynamic_pointer_cast<SpatialDomains::SegGeom>(
            lambda_make_edge(a, b)),
        std::dynamic_pointer_cast<SpatialDomains::SegGeom>(
            lambda_make_edge(b, c)),
        std::dynamic_pointer_cast<SpatialDomains::SegGeom>(
            lambda_make_edge(c, d)),
        std::dynamic_pointer_cast<SpatialDomains::SegGeom>(
            lambda_make_edge(d, a))};
    return std::dynamic_pointer_cast<Geometry2D>(
        std::make_shared<SpatialDomains::QuadGeom>(0, edges.data()));
  };

  std::vector<REAL> normal;

  {
    NekDouble a[3] = {0.0, 0.0, 0.0};
    NekDouble b[3] = {2.0, 0.0, 0.0};
    auto e = lambda_make_edge(a, b);
    get_normal_vector(e, normal);
    ASSERT_EQ(normal.size(), 2);
    ASSERT_NEAR(normal.at(0), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 1.0, 1.0e-15);

    get_vertex_average(std::static_pointer_cast<SpatialDomains::Geometry>(e),
                       normal);
    ASSERT_EQ(normal.size(), 3);
    ASSERT_NEAR(normal.at(0), 1.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(2), 0.0, 1.0e-15);
  }
  {
    NekDouble a[3] = {0.0, 0.0, 0.0};
    NekDouble b[3] = {0.0, -2.0, 0.0};
    auto e = lambda_make_edge(a, b);
    get_normal_vector(e, normal);
    ASSERT_EQ(normal.size(), 2);
    ASSERT_NEAR(normal.at(0), 1.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 0.0, 1.0e-15);

    get_vertex_average(std::static_pointer_cast<SpatialDomains::Geometry>(e),
                       normal);
    ASSERT_EQ(normal.size(), 3);
    ASSERT_NEAR(normal.at(0), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), -1.0, 1.0e-15);
    ASSERT_NEAR(normal.at(2), 0.0, 1.0e-15);
  }

  {
    NekDouble a[3] = {0.0, 0.0, 0.0};
    NekDouble b[3] = {1.0, 0.0, 0.0};
    NekDouble c[3] = {1.0, 1.0, 0.0};

    auto g = lambda_make_triangle(a, b, c);
    get_normal_vector(g, normal);
    ASSERT_EQ(normal.size(), 3);
    ASSERT_NEAR(normal.at(0), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(2), 1.0, 1.0e-15);

    get_vertex_average(std::static_pointer_cast<SpatialDomains::Geometry>(g),
                       normal);
    ASSERT_EQ(normal.size(), 3);
    ASSERT_NEAR(normal.at(0), 2.0 / 3.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 1.0 / 3.0, 1.0e-15);
    ASSERT_NEAR(normal.at(2), 0.0, 1.0e-15);
  }

  {
    NekDouble a[3] = {0.0, 0.0, 0.0};
    NekDouble b[3] = {0.0, 1.0, 0.0};
    NekDouble c[3] = {0.0, 1.0, 1.0};

    auto g = lambda_make_triangle(a, b, c);
    get_normal_vector(g, normal);
    ASSERT_EQ(normal.size(), 3);
    ASSERT_NEAR(normal.at(0), 1.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(2), 0.0, 1.0e-15);

    get_vertex_average(std::static_pointer_cast<SpatialDomains::Geometry>(g),
                       normal);
    ASSERT_EQ(normal.size(), 3);
    ASSERT_NEAR(normal.at(0), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 2.0 / 3.0, 1.0e-15);
    ASSERT_NEAR(normal.at(2), 1.0 / 3.0, 1.0e-15);
  }

  {
    NekDouble a[3] = {0.0, 0.0, 0.0};
    NekDouble b[3] = {1.0, 0.0, 0.0};
    NekDouble c[3] = {1.0, 1.0, 0.0};
    NekDouble d[3] = {0.0, 1.0, 0.0};

    auto g = lambda_make_quad(a, b, c, d);
    get_normal_vector(g, normal);
    ASSERT_EQ(normal.size(), 3);
    ASSERT_NEAR(normal.at(0), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(2), 1.0, 1.0e-15);

    get_vertex_average(std::static_pointer_cast<SpatialDomains::Geometry>(g),
                       normal);
    ASSERT_EQ(normal.size(), 3);
    ASSERT_NEAR(normal.at(0), 2.0 / 4.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 2.0 / 4.0, 1.0e-15);
    ASSERT_NEAR(normal.at(2), 0.0, 1.0e-15);
  }

  {
    NekDouble a[3] = {0.0, 0.0, 0.0};
    NekDouble b[3] = {0.0, 1.0, 0.0};
    NekDouble c[3] = {0.0, 1.0, 1.0};
    NekDouble d[3] = {0.0, 0.0, 1.0};

    auto g = lambda_make_quad(a, b, c, d);
    get_normal_vector(g, normal);
    ASSERT_EQ(normal.size(), 3);
    ASSERT_NEAR(normal.at(0), 1.0, 1.0e-15);
    ASSERT_NEAR(normal.at(1), 0.0, 1.0e-15);
    ASSERT_NEAR(normal.at(2), 0.0, 1.0e-15);
  }
}

TEST(CompositeInteraction, GeometryTransportAllD) {
  TestUtilities::TestResourceSession resources_session(
      "reference_all_types_cube/mixed_ref_cube_0.2.xml",
      "reference_all_types_cube/conditions.xml");
  auto session = resources_session.session;

  // Create MeshGraph.
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto comm = mesh->get_comm();
  auto sycl_target = std::make_shared<SYCLTarget>(0, comm);

  const int rank = sycl_target->comm_pair.rank_parent;

  auto lambda_compare_vertices = [&](auto A, auto B) {
    ASSERT_EQ(A->GetCoordim(), B->GetCoordim());
    ASSERT_EQ(A->GetGlobalID(), B->GetGlobalID());
    Nektar::NekDouble Ax = 0.0;
    Nektar::NekDouble Ay = 0.0;
    Nektar::NekDouble Az = 0.0;
    Nektar::NekDouble Bx = 0.0;
    Nektar::NekDouble By = 0.0;
    Nektar::NekDouble Bz = 0.0;
    A->GetCoords(Ax, Ay, Az);
    B->GetCoords(Bx, By, Bz);
    const int coordim = A->GetCoordim();
    if (coordim > 0) {
      ASSERT_NEAR(Ax, Bx, 1.0e-16);
    }
    if (coordim > 1) {
      ASSERT_NEAR(Ay, By, 1.0e-16);
    }
    if (coordim > 2) {
      ASSERT_NEAR(Az, Bz, 1.0e-16);
    }
  };

  auto lambda_compare_edges = [&](auto A, auto B) {
    ASSERT_EQ(A->GetCoordim(), B->GetCoordim());
    ASSERT_EQ(A->GetGlobalID(), B->GetGlobalID());

    ASSERT_EQ(A->GetNumVerts(), B->GetNumVerts());
    ASSERT_EQ(A->GetNumEdges(), B->GetNumEdges());
    ASSERT_EQ(A->GetNumFaces(), B->GetNumFaces());
    ASSERT_EQ(A->GetShapeDim(), B->GetShapeDim());

    const int num_vertices = A->GetNumVerts();
    for (int vx = 0; vx < num_vertices; vx++) {
      lambda_compare_vertices(A->GetVertex(vx), B->GetVertex(vx));
    }
  };

  for (auto &sg : graph->GetAllSegGeoms()) {
    GeometryTransport::RemoteGeom<SpatialDomains::SegGeom> rsg(rank, sg.first,
                                                               sg.second);
    GeometryTransport::RemoteGeom<SpatialDomains::SegGeom> rsgd;
    const std::size_t num_bytes = rsg.get_num_bytes();
    std::vector<std::byte> bytes(num_bytes);
    rsg.serialise(bytes.data(), num_bytes);
    rsgd.deserialise(bytes.data(), num_bytes);
    auto dg = rsgd.geom;
    ASSERT_TRUE(dg.get() != nullptr);
    lambda_compare_edges(sg.second, dg);
  }

  auto lambda_compare_faces = [&](auto A, auto B) {
    ASSERT_EQ(A->GetCoordim(), B->GetCoordim());
    ASSERT_EQ(A->GetGlobalID(), B->GetGlobalID());

    ASSERT_EQ(A->GetNumVerts(), B->GetNumVerts());
    ASSERT_EQ(A->GetNumEdges(), B->GetNumEdges());
    ASSERT_EQ(A->GetNumFaces(), B->GetNumFaces());
    ASSERT_EQ(A->GetShapeDim(), B->GetShapeDim());

    const int num_edges = A->GetNumEdges();
    for (int vx = 0; vx < num_edges; vx++) {
      lambda_compare_edges(A->GetEdge(vx), B->GetEdge(vx));
    }
  };

  auto lambda_check_face = [&](auto face_pair) {
    auto face_id = face_pair.first;
    auto face_geom = face_pair.second;
    GeometryTransport::RemoteGeom<SpatialDomains::Geometry2D> rsg(rank, face_id,
                                                                  face_geom);
    GeometryTransport::RemoteGeom<SpatialDomains::Geometry2D> rsgd;
    const std::size_t num_bytes = rsg.get_num_bytes();
    std::vector<std::byte> bytes(num_bytes);
    rsg.serialise(bytes.data(), num_bytes);
    rsgd.deserialise(bytes.data(), num_bytes);
    auto dg = rsgd.geom;
    ASSERT_TRUE(dg.get() != nullptr);
    lambda_compare_faces(face_geom, dg);
  };

  for (auto &sg : graph->GetAllTriGeoms()) {
    lambda_check_face(sg);
  }
  for (auto &sg : graph->GetAllQuadGeoms()) {
    lambda_check_face(sg);
  }

  auto lambda_compare_polys = [&](auto A, auto B) {
    ASSERT_EQ(A->GetCoordim(), B->GetCoordim());
    ASSERT_EQ(A->GetGlobalID(), B->GetGlobalID());

    ASSERT_EQ(A->GetNumVerts(), B->GetNumVerts());
    ASSERT_EQ(A->GetNumEdges(), B->GetNumEdges());
    ASSERT_EQ(A->GetNumFaces(), B->GetNumFaces());
    ASSERT_EQ(A->GetShapeDim(), B->GetShapeDim());

    const int num_faces = A->GetNumFaces();
    for (int vx = 0; vx < num_faces; vx++) {
      lambda_compare_faces(A->GetFace(vx), B->GetFace(vx));
    }
  };

  auto lambda_check_poly = [&](auto poly_pair) {
    auto poly_id = poly_pair.first;
    auto poly_geom = poly_pair.second;
    GeometryTransport::RemoteGeom<SpatialDomains::Geometry3D> rsg(rank, poly_id,
                                                                  poly_geom);
    GeometryTransport::RemoteGeom<SpatialDomains::Geometry3D> rsgd;
    const std::size_t num_bytes = rsg.get_num_bytes();
    std::vector<std::byte> bytes(num_bytes);
    rsg.serialise(bytes.data(), num_bytes);
    rsgd.deserialise(bytes.data(), num_bytes);
    auto dg = rsgd.geom;
    ASSERT_TRUE(dg.get() != nullptr);
    lambda_compare_polys(poly_geom, dg);
  };

  for (auto sg : graph->GetAllTetGeoms()) {
    lambda_check_poly(sg);
  }
  for (auto sg : graph->GetAllPyrGeoms()) {
    lambda_check_poly(sg);
  }
  for (auto sg : graph->GetAllPrismGeoms()) {
    lambda_check_poly(sg);
  }
  for (auto sg : graph->GetAllHexGeoms()) {
    lambda_check_poly(sg);
  }

  mesh->free();
  sycl_target->free();
}

class CompositeInteractionAllD
    : public testing::TestWithParam<std::tuple<std::string, std::string, int>> {
};
TEST_P(CompositeInteractionAllD, GeometryTransport) {

  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;

  // Create MeshGraph.
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto comm = mesh->get_comm();
  auto sycl_target = std::make_shared<SYCLTarget>(0, comm);

  std::vector<int> composite_indices = {100, 200, 300, 400, 500, 600};

  auto composite_transport =
      std::make_shared<CompositeTransportTester>(mesh, composite_indices);

  auto &contrib_cells = composite_transport->get_contrib_cells();

  int cell = -1;
  for (auto &ix : contrib_cells) {
    cell = ix;
  }

  int rank = sycl_target->comm_pair.rank_parent;
  int possible_rank = (cell == -1) ? -1 : rank;
  int chosen_rank;
  MPICHK(
      MPI_Allreduce(&possible_rank, &chosen_rank, 1, MPI_INT, MPI_MAX, comm));
  ASSERT_TRUE(chosen_rank >= 0);
  MPICHK(MPI_Bcast(&cell, 1, MPI_INT, chosen_rank, comm));

  std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
      remote_quads;
  std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
      remote_tris;
  std::vector<
      std::shared_ptr<GeometryTransport::RemoteGeom<SpatialDomains::SegGeom>>>
      remote_segments;
  std::set<INT> cells_set = {cell};

  const int num_collected = composite_transport->collect_geometry(cells_set);
  ASSERT_EQ(num_collected, 1);
  composite_transport->get_geometry(cell, remote_quads, remote_tris,
                                    remote_segments);

  std::vector<int> geom_int;
  std::vector<double> geom_real;

  auto lambda_push_data = [&](auto geom) {
    NekDouble x[3];
    geom_int.push_back(geom->GetCoordim());
    const int num_vertices = geom->GetNumVerts();
    for (int vx = 0; vx < num_vertices; vx++) {
      auto vert = geom->GetVertex(vx);
      vert->GetCoords(x[0], x[1], x[2]);
      for (int dx = 0; dx < ndim; dx++) {
        geom_real.push_back(x[dx]);
      }
    }
  };

  if (ndim == 3) {
    for (auto gx : remote_quads) {
      geom_int.push_back(gx->id);
      lambda_push_data(gx->geom);
    }
    for (auto gx : remote_tris) {
      geom_int.push_back(gx->id);
      lambda_push_data(gx->geom);
    }
  } else if (ndim == 2) {
    for (auto gx : remote_segments) {
      geom_int.push_back(gx->id);
      lambda_push_data(gx->geom);
    }
  }

  int num_int = geom_int.size();
  MPICHK(MPI_Bcast(&num_int, 1, MPI_INT, chosen_rank, comm));
  std::vector<int> geom_int_correct(num_int);
  if (rank == chosen_rank) {
    for (int ix = 0; ix < num_int; ix++) {
      geom_int_correct.at(ix) = geom_int.at(ix);
    }
  }
  MPICHK(
      MPI_Bcast(geom_int_correct.data(), num_int, MPI_INT, chosen_rank, comm));
  ASSERT_EQ(geom_int_correct, geom_int);

  int num_real = geom_real.size();
  MPICHK(MPI_Bcast(&num_real, 1, MPI_INT, chosen_rank, comm));
  std::vector<double> geom_real_correct(num_real);
  if (rank == chosen_rank) {
    for (int ix = 0; ix < num_real; ix++) {
      geom_real_correct.at(ix) = geom_real.at(ix);
    }
  }
  MPICHK(MPI_Bcast(geom_real_correct.data(), num_real, MPI_DOUBLE, chosen_rank,
                   comm));
  ASSERT_EQ(geom_real_correct, geom_real);

  composite_transport->free();
  mesh->free();
}

TEST_P(CompositeInteractionAllD, Collections) {
  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;

  // Create MeshGraph.
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto comm = mesh->get_comm();
  auto sycl_target = std::make_shared<SYCLTarget>(0, comm);

  std::vector<int> composite_indices = {100, 200, 300, 400, 500, 600};
  std::map<int, std::vector<int>> boundary_groups = {{1, composite_indices}};

  auto composite_transport =
      std::make_shared<CompositeTransportTester>(mesh, composite_indices);

  auto &contrib_cells = composite_transport->get_contrib_cells();

  int cell = -1;
  for (auto &ix : contrib_cells) {
    cell = ix;
  }

  int rank = sycl_target->comm_pair.rank_parent;
  int possible_rank = (cell == -1) ? -1 : rank;
  int chosen_rank;
  MPICHK(
      MPI_Allreduce(&possible_rank, &chosen_rank, 1, MPI_INT, MPI_MAX, comm));
  ASSERT_TRUE(chosen_rank >= 0);
  MPICHK(MPI_Bcast(&cell, 1, MPI_INT, chosen_rank, comm));

  std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
      remote_quads;
  std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
      remote_tris;

  std::set<INT> cells_set = {cell};

  const int num_collected = composite_transport->collect_geometry(cells_set);
  ASSERT_EQ(num_collected, 1);
  std::vector<
      std::shared_ptr<GeometryTransport::RemoteGeom<SpatialDomains::SegGeom>>>
      remote_segments;
  composite_transport->get_geometry(cell, remote_quads, remote_tris,
                                    remote_segments);

  std::set<INT> cell_arg;
  cell_arg.insert(cell);
  auto composite_collections = std::make_shared<CompositeCollections>(
      sycl_target, mesh, boundary_groups);
  composite_collections->collect_geometry(cell_arg);

  auto map_cells_collections = composite_collections->map_cells_collections;
  CompositeCollection *d_cc;
  CompositeCollection h_cc;
  auto exists = map_cells_collections->host_get(cell, &d_cc);
  ASSERT_TRUE(exists);
  sycl_target->queue.memcpy(&h_cc, d_cc, sizeof(CompositeCollection))
      .wait_and_throw();

  auto q = sycl_target->queue;
  if (ndim == 3) {

    int correct_num_quads;
    int correct_num_tris;

    if (rank == chosen_rank) {
      correct_num_quads = h_cc.num_quads;
      ASSERT_EQ(correct_num_quads, remote_quads.size());
      correct_num_tris = h_cc.num_tris;
      ASSERT_EQ(correct_num_tris, remote_tris.size());
    }

    MPICHK(MPI_Bcast(&correct_num_quads, 1, MPI_INT, chosen_rank, comm));
    MPICHK(MPI_Bcast(&correct_num_tris, 1, MPI_INT, chosen_rank, comm));

    EXPECT_EQ(h_cc.num_quads, correct_num_quads);
    EXPECT_EQ(h_cc.num_tris, correct_num_tris);

    std::vector<int> correct_composite_ids_quads(correct_num_quads);
    std::vector<int> correct_composite_ids_tris(correct_num_tris);
    std::vector<int> correct_geom_ids_quads(correct_num_quads);
    std::vector<int> correct_geom_ids_tris(correct_num_tris);
    std::vector<int> test_composite_ids_quads(correct_num_quads);
    std::vector<int> test_composite_ids_tris(correct_num_tris);
    std::vector<int> test_geom_ids_quads(correct_num_quads);
    std::vector<int> test_geom_ids_tris(correct_num_tris);

    if (rank == chosen_rank) {
      q.memcpy(correct_composite_ids_quads.data(), h_cc.composite_ids_quads,
               correct_num_quads * sizeof(int))
          .wait_and_throw();
      q.memcpy(correct_composite_ids_tris.data(), h_cc.composite_ids_tris,
               correct_num_tris * sizeof(int))
          .wait_and_throw();
      q.memcpy(correct_geom_ids_quads.data(), h_cc.geom_ids_quads,
               correct_num_quads * sizeof(int))
          .wait_and_throw();
      q.memcpy(correct_geom_ids_tris.data(), h_cc.geom_ids_tris,
               correct_num_tris * sizeof(int))
          .wait_and_throw();

      std::vector<CompositeInteraction::LinePlaneIntersection> quads_lpi(
          correct_num_quads);
      std::vector<CompositeInteraction::LinePlaneIntersection> tris_lpi(
          correct_num_tris);
      q.memcpy(quads_lpi.data(), h_cc.lpi_quads,
               correct_num_quads *
                   sizeof(CompositeInteraction::LinePlaneIntersection))
          .wait_and_throw();
      q.memcpy(tris_lpi.data(), h_cc.lpi_tris,
               correct_num_tris *
                   sizeof(CompositeInteraction::LinePlaneIntersection))
          .wait_and_throw();

      auto lambda_find_geom = [&](const int gid, auto container) -> int {
        int index = 0;
        for (auto gx : container) {
          if (gx->id == gid) {
            return index;
          }
          index++;
        }
        return -1;
      };
      for (int qx = 0; qx < correct_num_quads; qx++) {
        const int gid = correct_geom_ids_quads.at(qx);
        const int index = lambda_find_geom(gid, remote_quads);
        ASSERT_EQ(remote_quads.at(index)->id, gid);
        auto rgeom = remote_quads.at(index);
        CompositeInteraction::LinePlaneIntersection lpi_correct(rgeom->geom);
        auto lpi_to_test = quads_lpi.at(qx);
        ASSERT_NEAR(lpi_correct.point0, lpi_to_test.point0, 1.0e-15);
        ASSERT_NEAR(lpi_correct.point1, lpi_to_test.point1, 1.0e-15);
        ASSERT_NEAR(lpi_correct.point2, lpi_to_test.point2, 1.0e-15);
        ASSERT_NEAR(lpi_correct.normal0, lpi_to_test.normal0, 1.0e-15);
        ASSERT_NEAR(lpi_correct.normal1, lpi_to_test.normal1, 1.0e-15);
        ASSERT_NEAR(lpi_correct.normal2, lpi_to_test.normal2, 1.0e-15);
      }
      for (int qx = 0; qx < correct_num_tris; qx++) {
        const int gid = correct_geom_ids_tris.at(qx);
        const int index = lambda_find_geom(gid, remote_tris);
        ASSERT_EQ(remote_tris.at(index)->id, gid);
        auto rgeom = remote_tris.at(index);
        CompositeInteraction::LinePlaneIntersection lpi_correct(rgeom->geom);
        auto lpi_to_test = tris_lpi.at(qx);
        ASSERT_NEAR(lpi_correct.point0, lpi_to_test.point0, 1.0e-15);
        ASSERT_NEAR(lpi_correct.point1, lpi_to_test.point1, 1.0e-15);
        ASSERT_NEAR(lpi_correct.point2, lpi_to_test.point2, 1.0e-15);
        ASSERT_NEAR(lpi_correct.normal0, lpi_to_test.normal0, 1.0e-15);
        ASSERT_NEAR(lpi_correct.normal1, lpi_to_test.normal1, 1.0e-15);
        ASSERT_NEAR(lpi_correct.normal2, lpi_to_test.normal2, 1.0e-15);
      }
    }
    q.memcpy(test_composite_ids_quads.data(), h_cc.composite_ids_quads,
             correct_num_quads * sizeof(int))
        .wait_and_throw();
    q.memcpy(test_composite_ids_tris.data(), h_cc.composite_ids_tris,
             correct_num_tris * sizeof(int))
        .wait_and_throw();
    q.memcpy(test_geom_ids_quads.data(), h_cc.geom_ids_quads,
             correct_num_quads * sizeof(int))
        .wait_and_throw();
    q.memcpy(test_geom_ids_tris.data(), h_cc.geom_ids_tris,
             correct_num_tris * sizeof(int))
        .wait_and_throw();

    MPICHK(MPI_Bcast(correct_composite_ids_quads.data(), correct_num_quads,
                     MPI_INT, chosen_rank, comm));
    MPICHK(MPI_Bcast(correct_composite_ids_tris.data(), correct_num_tris,
                     MPI_INT, chosen_rank, comm));
    MPICHK(MPI_Bcast(correct_geom_ids_quads.data(), correct_num_quads, MPI_INT,
                     chosen_rank, comm));
    MPICHK(MPI_Bcast(correct_geom_ids_tris.data(), correct_num_tris, MPI_INT,
                     chosen_rank, comm));

    EXPECT_EQ(correct_composite_ids_quads, test_composite_ids_quads);
    EXPECT_EQ(correct_composite_ids_tris, test_composite_ids_tris);
    EXPECT_EQ(correct_geom_ids_quads, test_geom_ids_quads);
    EXPECT_EQ(correct_geom_ids_tris, test_geom_ids_tris);

  } else if (ndim == 2) {

    int correct_num_segments;

    if (rank == chosen_rank) {
      correct_num_segments = h_cc.num_segments;
      ASSERT_EQ(correct_num_segments, remote_segments.size());
    }

    MPICHK(MPI_Bcast(&correct_num_segments, 1, MPI_INT, chosen_rank, comm));
    EXPECT_EQ(h_cc.num_segments, correct_num_segments);

    std::vector<int> test_composite_ids(correct_num_segments);
    std::vector<int> test_geom_ids(correct_num_segments);
    std::vector<int> correct_composite_ids(correct_num_segments);
    std::vector<int> correct_geom_ids(correct_num_segments);

    q.memcpy(test_composite_ids.data(), h_cc.composite_ids_segments,
             correct_num_segments * sizeof(int))
        .wait_and_throw();
    q.memcpy(test_geom_ids.data(), h_cc.geom_ids_segments,
             correct_num_segments * sizeof(int))
        .wait_and_throw();

    if (rank == chosen_rank) {
      q.memcpy(correct_composite_ids.data(), h_cc.composite_ids_segments,
               correct_num_segments * sizeof(int))
          .wait_and_throw();
      q.memcpy(correct_geom_ids.data(), h_cc.geom_ids_segments,
               correct_num_segments * sizeof(int))
          .wait_and_throw();
    }

    MPICHK(MPI_Bcast(correct_composite_ids.data(), correct_num_segments,
                     MPI_INT, chosen_rank, comm));
    MPICHK(MPI_Bcast(correct_geom_ids.data(), correct_num_segments, MPI_INT,
                     chosen_rank, comm));
    EXPECT_EQ(correct_composite_ids, test_composite_ids);
    EXPECT_EQ(correct_geom_ids, test_geom_ids);
  }

  composite_transport->free();
  composite_collections->free();
  sycl_target->free();
  mesh->free();
}

TEST_P(CompositeInteractionAllD, Intersection) {
  const int N_total = 5000;
  NekDouble newton_tol = 1.0e-8;

  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;

  // Create MeshGraph.
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());
  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("NORMAL"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;
  std::mt19937 rng_pos(52234234 + rank);
  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  int N = rend - rstart;
  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");
  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
    }
    A->add_particles_local(initial_distribution);
  }

  pbc.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[100] = {100};
  boundary_groups[200] = {200};
  boundary_groups[300] = {300};
  boundary_groups[400] = {400};
  if (ndim > 2) {
    boundary_groups[500] = {500};
    boundary_groups[600] = {600};
  }

  auto composite_intersection = std::make_shared<CompositeIntersectionTester>(
      sycl_target, mesh, boundary_groups);

  // Test pre integration actually copied the current positions
  composite_intersection->pre_integration(A);
  if (!A->contains_dat(Sym<REAL>("NESO_COMP_INT_OUTPUT_POS"))) {
    A->add_particle_dat(Sym<REAL>("NESO_COMP_INT_OUTPUT_POS"), ndim);
  }
  if (!A->contains_dat(Sym<INT>("NESO_COMP_INT_OUTPUT_COMP"))) {
    A->add_particle_dat(Sym<INT>("NESO_COMP_INT_OUTPUT_COMP"), 2);
  }

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto P = A->position_dat->cell_dat.get_cell(cellx);
    auto PP = A->get_dat(composite_intersection->previous_position_sym)
                  ->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < P->nrow; rowx++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        ASSERT_EQ((*P)[dimx][rowx], (*PP)[dimx][rowx]);
      }
    }
  }

  // find cells on the unmoved particles should return the mesh hierarchy cells
  // the particles are currently in
  std::set<INT> cells;
  composite_intersection->test_find_cells(A, cells);

  auto mesh_hierarchy_mapper = std::make_unique<MeshHierarchyMapper>(
      sycl_target, mesh->get_mesh_hierarchy());
  const auto mesh_hierarchy_device_mapper =
      mesh_hierarchy_mapper->get_host_mapper();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto P = A->position_dat->cell_dat.get_cell(cellx);
    for (int rowx = 0; rowx < P->nrow; rowx++) {
      REAL position[3] = {0.0, 0.0, 0.0};
      INT mh_cell[6] = {0, 0, 0, 0, 0, 0};
      for (int dimx = 0; dimx < ndim; dimx++) {
        position[dimx] = (*P)[dimx][rowx];
      }
      mesh_hierarchy_device_mapper.map_to_tuple(position, mh_cell);
      const INT linear_cell =
          mesh_hierarchy_device_mapper.tuple_to_linear_global(mh_cell);
      ASSERT_TRUE(cells.count(linear_cell));
    }
  }

  REAL offset_x;
  REAL offset_y;
  REAL offset_z;

  auto lambda_apply_offset = [&]() {
    particle_loop(
        A,
        [=](auto P) {
          P.at(0) += offset_x;
          P.at(1) += offset_y;
          if (ndim > 2) {
            P.at(2) += offset_z;
          }
        },
        Access::write(Sym<REAL>("P")))
        ->execute();
  };

  auto reset_positions = particle_loop(
      A,
      [=](auto P, auto PP) {
        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) = PP.at(dx);
        }
      },
      Access::write(Sym<REAL>("P")),
      Access::read(Sym<REAL>("NESO_COMP_INT_PREV_POS")));

  auto lambda_test_normal = [&](auto correct_normal, auto &sub_groups) {
    auto device_normal_mapper = composite_intersection->composite_collections
                                    ->get_device_normal_mapper();
    auto error_propagate = std::make_shared<ErrorPropagate>(sycl_target);

    for (auto &pairx : sub_groups) {
      auto k_ep = error_propagate->device_ptr();

      particle_loop(
          pairx.second,
          [=](auto IN, auto IC) {
            REAL *normal;
            const INT geom_id = IC.at_ephemeral(1);
            const bool has_normal = device_normal_mapper.get(geom_id, &normal);
            NESO_KERNEL_ASSERT(has_normal, k_ep);
            for (int dx = 0; dx < ndim; dx++) {
              IN.at(dx) = normal[dx];
            }
          },
          Access::write(Sym<REAL>("NORMAL")),
          Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")))
          ->execute();
      ASSERT_FALSE(error_propagate->get_flag());
    }

    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto IN = A->get_cell(Sym<REAL>("NORMAL"), cellx);
      for (int rowx = 0; rowx < IN->nrow; rowx++) {
        for (int dx = 0; dx < ndim; dx++) {
          ASSERT_TRUE(
              (std::abs(IN->at(rowx, dx) - correct_normal[dx]) < 1.0e-15) ||
              (std::abs(IN->at(rowx, dx) + correct_normal[dx]) < 1.0e-15));
        }
      }
    }
  };

  auto lambda_test = [&](const int expected_composite, auto correct_normal) {
    composite_intersection->pre_integration(A);

    ASSERT_TRUE(A->contains_dat(Sym<REAL>("NESO_COMP_INT_PREV_POS")));
    lambda_apply_offset();
    auto sub_groups = composite_intersection->get_intersections(A);

    for (auto &pairx : sub_groups) {

      ASSERT_TRUE(pairx.second->contains_ephemeral_dat(
          Sym<REAL>("NESO_PARTICLES_BOUNDARY_INTERSECTION_POINT")));
      ASSERT_TRUE(pairx.second->contains_ephemeral_dat(
          Sym<REAL>("NESO_PARTICLES_BOUNDARY_NORMAL")));
      ASSERT_TRUE(pairx.second->contains_ephemeral_dat(
          Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")));

      particle_loop(
          pairx.second,
          [=](auto OUTPUT_POS, auto OUTPUT_COMP, auto EPH_POS, auto EPH_COMP) {
            for (int dx = 0; dx < ndim; dx++) {
              OUTPUT_POS.at(dx) = EPH_POS.at_ephemeral(dx);
            }
            OUTPUT_COMP.at(0) = EPH_COMP.at_ephemeral(0);
            OUTPUT_COMP.at(1) = EPH_COMP.at_ephemeral(1);
          },
          Access::write(Sym<REAL>("NESO_COMP_INT_OUTPUT_POS")),
          Access::write(Sym<INT>("NESO_COMP_INT_OUTPUT_COMP")),
          Access::read(Sym<REAL>("NESO_PARTICLES_BOUNDARY_INTERSECTION_POINT")),
          Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")))
          ->execute();
    }

    int local_count = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto P = A->get_cell(Sym<REAL>("P"), cellx);
      auto IP = A->get_cell(Sym<REAL>("NESO_COMP_INT_OUTPUT_POS"), cellx);
      auto IC = A->get_cell(Sym<INT>("NESO_COMP_INT_OUTPUT_COMP"), cellx);
      for (int rowx = 0; rowx < P->nrow; rowx++) {

        auto hit_composite = IC->at(rowx, 0);
        auto geom_id = IC->at(rowx, 1);
        auto composite_id = composite_intersection->composite_collections
                                ->map_geom_id_to_composite_id.at(geom_id);
        ASSERT_EQ(hit_composite, expected_composite);

        auto geom = composite_intersection->composite_collections
                        ->map_geom_id_to_geoms.at(geom_id);

        Array<OneD, NekDouble> point(3);
        Array<OneD, NekDouble> local_point(3);
        Array<OneD, NekDouble> global_point(3);

        point[0] = IP->at(rowx, 0);
        point[1] = IP->at(rowx, 1);
        if (ndim == 3) {
          point[2] = IP->at(rowx, 2);
        } else {
          point[2] = 0.0;
        }

        NekDouble dist;
        bool contained =
            geom->ContainsPoint(point, local_point, newton_tol, dist);
        if (!contained) {
          geom->GetLocCoords(point, local_point);
          for (int dx = 0; dx < ndim; dx++) {
            global_point[dx] = geom->GetCoord(dx, local_point);
          }
          double dist = 0.0;
          for (int dx = 0; dx < ndim; dx++) {
            const double r = point[dx] - global_point[dx];
            dist += r * r;
          }
          dist = std::sqrt(dist);
          contained = dist < newton_tol * 10.0;
        }
        ASSERT_TRUE(contained);
        local_count++;
      }
    }

    for (const auto bgx : boundary_groups) {
      const int cx = bgx.first;
      if (cx == expected_composite) {
        ASSERT_EQ(sub_groups.at(cx)->get_npart_local(), local_count);
      } else {
        ASSERT_EQ(sub_groups.at(cx)->get_npart_local(), 0);
      }
    }

    lambda_test_normal(correct_normal, sub_groups);
  };

  REAL correct_normal[3] = {0.0, 0.0, 0.0};
  if (ndim == 2) {
    offset_x = 0.0;
    offset_y = 4.0;
    offset_z = 0.0;
    correct_normal[0] = 0.0;
    correct_normal[1] = 1.0;
    correct_normal[2] = 0.0;
    lambda_test(300, correct_normal);
    reset_positions->execute();
    offset_x = -4.0;
    offset_y = 0.0;
    offset_z = 0.0;
    correct_normal[0] = 1.0;
    correct_normal[1] = 0.0;
    correct_normal[2] = 0.0;
    lambda_test(400, correct_normal);
    reset_positions->execute();
    offset_x = 4.0;
    offset_y = 0.0;
    offset_z = 0.0;
    correct_normal[0] = 1.0;
    correct_normal[1] = 0.0;
    correct_normal[2] = 0.0;
    lambda_test(200, correct_normal);
    reset_positions->execute();
    offset_x = 0.0;
    offset_y = -4.0;
    offset_z = 0.0;
    correct_normal[0] = 0.0;
    correct_normal[1] = 1.0;
    correct_normal[2] = 0.0;
    lambda_test(100, correct_normal);
    reset_positions->execute();
  } else if (ndim == 3) {
    offset_x = 2.0;
    offset_y = 0.0;
    offset_z = 0.0;
    correct_normal[0] = 1.0;
    correct_normal[1] = 0.0;
    correct_normal[2] = 0.0;
    lambda_test(300, correct_normal);
    reset_positions->execute();
    offset_x = -2.0;
    offset_y = 0.0;
    offset_z = 0.0;
    correct_normal[0] = 1.0;
    correct_normal[1] = 0.0;
    correct_normal[2] = 0.0;
    lambda_test(400, correct_normal);
    reset_positions->execute();
    offset_x = 0.0;
    offset_y = 2.0;
    offset_z = 0.0;
    correct_normal[0] = 0.0;
    correct_normal[1] = 1.0;
    correct_normal[2] = 0.0;
    lambda_test(200, correct_normal);
    reset_positions->execute();
    offset_x = 0.0;
    offset_y = -2.0;
    offset_z = 0.0;
    correct_normal[0] = 0.0;
    correct_normal[1] = 1.0;
    correct_normal[2] = 0.0;
    lambda_test(100, correct_normal);
    reset_positions->execute();
    offset_x = 0.0;
    offset_y = 0.0;
    offset_z = 2.0;
    correct_normal[0] = 0.0;
    correct_normal[1] = 0.0;
    correct_normal[2] = 1.0;
    lambda_test(600, correct_normal);
    reset_positions->execute();
    offset_x = 0.0;
    offset_y = 0.0;
    offset_z = -2.0;
    correct_normal[0] = 0.0;
    correct_normal[1] = 0.0;
    correct_normal[2] = 1.0;
    lambda_test(500, correct_normal);
    reset_positions->execute();
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST_P(CompositeInteractionAllD, Reflection) {
  const int N_total = 4000;
  const REAL dt = 0.05;
  const int N_steps = 50;

  std::tuple<std::string, std::string, double> param = GetParam();

  const std::string filename_conditions = std::get<0>(param);
  const std::string filename_mesh = std::get<1>(param);
  const int ndim = std::get<2>(param);

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;

  // Create MeshGraph.
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto config = std::make_shared<ParameterStore>();
  config->set<REAL>("MapParticles3DRegular/tol", 1.0e-10);
  config->set<REAL>("CompositeIntersection/newton_tol", 1.0e-8);
  config->set<REAL>("CompositeIntersection/line_intersection_tol", 1.0e-10);
  config->set<REAL>("NektarCompositeTruncatedReflection/reset_distance",
                    1.0e-6);

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh, config);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<REAL>("TSP"), 2),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;
  std::mt19937 rng_pos(52234234 + rank);
  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  int N = rend - rstart;
  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");
  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    auto velocities =
        NESO::Particles::normal_distribution(N, 3, 0.0, 0.5, rng_pos);

    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
        initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    A->add_particles_local(initial_distribution);
  }

  pbc.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  std::vector<int> reflection_composite_indices = {100, 200, 300, 400};
  if (ndim > 2) {
    reflection_composite_indices.push_back(500);
    reflection_composite_indices.push_back(600);
  }

  auto reflection = std::make_shared<NektarCompositeTruncatedReflection>(
      Sym<REAL>("V"), Sym<REAL>("TSP"), sycl_target, mesh,
      reflection_composite_indices, config);

  auto lambda_apply_timestep_reset = [&](auto aa) {
    particle_loop(
        aa,
        [=](auto TSP) {
          TSP.at(0) = 0.0;
          TSP.at(1) = 0.0;
        },
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };

  auto lambda_apply_advection_step =
      [=](ParticleSubGroupSharedPtr iteration_set) -> void {
    particle_loop(
        "euler_advection", iteration_set,
        [=](auto V, auto P, auto TSP) {
          const REAL dt_left = dt - TSP.at(0);
          if (dt_left > 0.0) {
            for (int dx = 0; dx < ndim; dx++) {
              P.at(dx) += dt_left * V.at(dx);
            }
            TSP.at(0) = dt;
            TSP.at(1) = dt_left;
          }
        },
        Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("P")),
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };

  auto lambda_pre_advection = [&](auto aa) { reflection->pre_advection(aa); };

  auto lambda_apply_boundary_conditions = [&](auto aa) {
    reflection->execute(aa);
  };

  auto lambda_find_partial_moves = [&](auto aa) {
    return static_particle_sub_group(
        A, [=](auto TSP) { return TSP.at(0) < dt; },
        Access::read(Sym<REAL>("TSP")));
  };

  auto lambda_partial_moves_remaining = [&](auto aa) -> bool {
    const int size = aa->get_npart_local();
    int size_global;
    MPICHK(MPI_Allreduce(&size, &size_global, 1, MPI_INT, MPI_SUM,
                         sycl_target->comm_pair.comm_parent));
    return size_global > 0;
  };

  auto lambda_apply_timestep = [&](auto aa) {
    lambda_apply_timestep_reset(aa);
    lambda_pre_advection(aa);
    lambda_apply_advection_step(aa);
    lambda_apply_boundary_conditions(aa);
    aa = lambda_find_partial_moves(aa);
    while (lambda_partial_moves_remaining(aa)) {
      lambda_pre_advection(aa);
      lambda_apply_advection_step(aa);
      lambda_apply_boundary_conditions(aa);
      aa = lambda_find_partial_moves(aa);
    }
  };

  auto error_propagate = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = error_propagate->device_ptr();
  auto check_loop = particle_loop(
      A,
      [=](auto TSP) {
        const bool cond = std::abs(TSP.at(0) - dt) < 1.0e-15;
        NESO_KERNEL_ASSERT(cond, k_ep);
      },
      Access::read(Sym<REAL>("TSP")));

  for (int stepx = 0; stepx < N_steps; stepx++) {
    lambda_apply_timestep(static_particle_sub_group(A));
    check_loop->execute();
    EXPECT_TRUE(!error_propagate->get_flag());
    A->hybrid_move();
    cell_id_translation->execute();
    A->cell_move();
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

INSTANTIATE_TEST_SUITE_P(
    MultipleMeshes, CompositeInteractionAllD,
    testing::Values(std::tuple<std::string, std::string, int>(
                        "conditions.xml", "square_triangles_quads.xml", 2),
                    std::tuple<std::string, std::string, double>(
                        "reference_all_types_cube/conditions.xml",
                        "reference_all_types_cube/linear_non_regular_0.5.xml",
                        3)));
