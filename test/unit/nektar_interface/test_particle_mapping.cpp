#include "nektar_interface/particle_interface.hpp"
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
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

// Test advecting particles between ranks
TEST(ParticleGeometryInterface, LocalMapping) {

  const int N_total = 2000;
  const double tol = 1.0e-10;
  int argc = 2;
  char *argv[2];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path mesh_file =
      test_resources_dir / "square_triangles_quads.xml";
  copy_to_cstring(std::string(mesh_file), &argv[1]);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraph::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());










  auto quad0 = mesh->graph->GetAllQuadGeoms().begin()->second;


  
  Array<OneD, NekDouble> coords(3);
  Array<OneD, NekDouble> Lcoords(3);


  nprint("--------------------------------------------");
  coords[0] = 0.0; coords[1] = 0.0;
  quad0->GetLocCoords(coords, Lcoords);
  nprint(coords[0], coords[1], "->", Lcoords[0], Lcoords[1]);
  coords[0] = 1.0; coords[1] = 0.0;
  quad0->GetLocCoords(coords, Lcoords);
  nprint(coords[0], coords[1], "->", Lcoords[0], Lcoords[1]);
  coords[0] = 0.0; coords[1] = 1.0;
  quad0->GetLocCoords(coords, Lcoords);
  nprint(coords[0], coords[1], "->", Lcoords[0], Lcoords[1]);
  nprint("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
  

  
  double shift[2];
  Lcoords[0] = 0.0;
  Lcoords[1] = 0.0;
  shift[0] = quad0->GetCoord(0, Lcoords);
  shift[1] = quad0->GetCoord(1, Lcoords);
  
  nprint("shift", shift[0], shift[1]);

  double matrix[4];
  Lcoords[0] = 1.0;
  Lcoords[1] = 0.0;
  matrix[0] = quad0->GetCoord(0, Lcoords) - shift[0];
  matrix[2] = quad0->GetCoord(1, Lcoords) - shift[1];

  Lcoords[0] = 0.0;
  Lcoords[1] = 1.0;
  matrix[1] = quad0->GetCoord(0, Lcoords) - shift[0];
  matrix[3] = quad0->GetCoord(1, Lcoords) - shift[1];
  
  nprint("M start");
  nprint(matrix[0], matrix[1]);
  nprint(matrix[2], matrix[3]);
  nprint("M end");

  Lcoords[0] = 0.5;
  Lcoords[1] = 0.0;
  nprint("test start");
  nprint(Lcoords[0], Lcoords[1], quad0->GetCoord(0, Lcoords));
  nprint(Lcoords[0], Lcoords[1], quad0->GetCoord(1, Lcoords));
  nprint("test end");


  Lcoords[0] = 0.0;
  Lcoords[1] = 0.5;
  nprint("test start");
  nprint(Lcoords[0], Lcoords[1], quad0->GetCoord(0, Lcoords));
  nprint(Lcoords[0], Lcoords[1], quad0->GetCoord(1, Lcoords));
  nprint("test end");



  nprint("............................................");


  double inverse_matrix[4] = {0.0, 16.0, -20.0, 0.0};

  
  auto lambda_map_to_ref = [&] (double * in, double * out) {
    double tmp[2];

    tmp[0] = in[0] - shift[0];
    tmp[1] = in[1] - shift[1];

    out[0] = inverse_matrix[0] * tmp[0] + inverse_matrix[1] * tmp[1];
    out[1] = inverse_matrix[2] * tmp[0] + inverse_matrix[3] * tmp[1];

  };

  double test_in[2];
  double test_out[2];

  test_in[0] = 0.0;
  test_in[1] = 0.0;
  lambda_map_to_ref(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);

  test_in[0] = 1.0;
  test_in[1] = 0.0;
  lambda_map_to_ref(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);

  test_in[0] = 0.0;
  test_in[1] = 1.0;
  lambda_map_to_ref(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);


  nprint("............................................");

  auto quad_geom_factors = quad0->GetGeomFactors();
  auto foo0 = quad0->GetCoeffs(0);
  nprint("foo0.size()", foo0.size());
  nprint("foo0:", foo0[0], foo0[1], foo0[2], foo0[3]);
  auto foo1 = quad0->GetCoeffs(1);
  nprint("foo1.size()", foo1.size());
  nprint("foo1:", foo1[0], foo1[1], foo1[2], foo1[3]);

   auto lambda_map_to_phys_quad = [&] (double * in, double * out) {
    const double x0 = in[0];
    const double x1 = in[1];
    const double c0 = (1.0 - x0) * (1.0 - x1) * 0.25;
    const double c1 = (1.0 + x0) * (1.0 - x1) * 0.25;
    const double c2 = (1.0 - x0) * (1.0 + x1) * 0.25;
    const double c3 = (1.0 + x0) * (1.0 + x1) * 0.25;
    
    out[0] = foo0[0] * c0 + foo0[1] * c1 + foo0[2] * c2 + foo0[3] * c3;
    out[1] = foo1[0] * c0 + foo1[1] * c1 + foo1[2] * c2 + foo1[3] * c3;

  }; 
  
  
  test_in[0] = 0.0;
  test_in[1] = 0.0;
  lambda_map_to_phys_quad(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);

  test_in[0] = 3.0;
  test_in[1] = 19.0;
  lambda_map_to_phys_quad(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);

  test_in[0] = 3.0;
  test_in[1] = -1.0;
  lambda_map_to_phys_quad(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);

  test_in[0] = 19.0;
  test_in[1] = 19.0;
  lambda_map_to_phys_quad(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);


  nprint("============================================");



  auto lambda_inverse_n_map = [&] (
    double *eta, double * xi)
  {
    xi[0] = (1.0 + eta[0]) * (1.0 - eta[1]) * 0.5 - 1.0;
    xi[1] = eta[1];

    //xi[0] = eta[0];
    //xi[1] = eta[1];
  };


  auto lambda_n_map = [&] (double * xi, double * eta) {
    NekDouble d1 = 1. - xi[1];
    if (fabs(d1) < NekConstants::kNekZeroTol)
    {
        if (d1 >= 0.)
        {
            d1 = NekConstants::kNekZeroTol;
        }
        else
        {
            d1 = -NekConstants::kNekZeroTol;
        }
    }
    eta[0] = 2. * (1. + xi[0]) / d1 - 1.0;
    eta[1] = xi[1];


    //eta[0] = xi[0];
    //eta[1] = xi[1];
  }; 


  auto tri0 = mesh->graph->GetAllTriGeoms().begin()->second;
 

  double eta[2];
  double xi[2];

  xi[0] = 0.0;
  xi[1] = 0.0;
  lambda_n_map(xi, eta);

  double shift_tri[2];
  double matrix_tri[4];

  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  shift_tri[0] = tri0->GetCoord(0, Lcoords);
  shift_tri[1] = tri0->GetCoord(1, Lcoords);

  nprint("shift tri:", shift_tri[0], shift_tri[1]);


  xi[0] = -1.0;
  xi[1] = 0.0;
  lambda_n_map(xi, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  matrix_tri[0] = -1.0 * (tri0->GetCoord(0, Lcoords) - shift_tri[0]);
  matrix_tri[2] = -1.0 * (tri0->GetCoord(1, Lcoords) - shift_tri[1]);

  xi[0] = 0.0;
  xi[1] = -0.5;
  lambda_n_map(xi, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  lambda_inverse_n_map(eta, xi);
  nprint("y eta", eta[0], eta[1], "xi", xi[0], xi[1]);
  matrix_tri[1] = -2.0 * (tri0->GetCoord(0, Lcoords) - shift_tri[0]);
  matrix_tri[3] = -2.0 * (tri0->GetCoord(1, Lcoords) - shift_tri[1]);


  nprint("M tri start");
  nprint(matrix_tri[0], matrix_tri[1]);
  nprint(matrix_tri[2], matrix_tri[3]);
  nprint("np.array(((", matrix_tri[0], "," ,matrix_tri[1], "),(", matrix_tri[2], "," ,matrix_tri[3], ")))" );
  nprint("M tri end");
  
  //-1
  //double inverse_matrix_tri[4] = {1.79816239e+01,  1.06666667e+01, -2.14140099e+01,  5.07186542e-14};
  // -0.5
  double inverse_matrix_tri[4] = {2.15506256e+01,  1.06666667e+01, -2.14140099e+01,  5.07186542e-14};

  auto lambda_map_to_ref_tri = [&] (double * in, double * out) {
    double tmp[2];
    double tmp_xi[2];

    tmp[0] = in[0] - shift_tri[0];
    tmp[1] = in[1] - shift_tri[1];

    tmp_xi[0] = inverse_matrix_tri[0] * tmp[0] + inverse_matrix_tri[1] * tmp[1];
    tmp_xi[1] = inverse_matrix_tri[2] * tmp[0] + inverse_matrix_tri[3] * tmp[1];

    lambda_n_map(tmp_xi, out);
  };

  auto lambda_map_to_phys_tri = [&] (double * in, double * out) {
    double tmp_xi[2];

    lambda_inverse_n_map(in, tmp_xi);
    out[0] = matrix_tri[0] * tmp_xi[0] + matrix_tri[1] * tmp_xi[1];
    out[1] = matrix_tri[2] * tmp_xi[0] + matrix_tri[3] * tmp_xi[1];
    out[0] += shift_tri[0];
    out[1] += shift_tri[1];

  };

  double ref[2];
  double phys[2];
  double test_phys[2];

  Lcoords[0] = 0.0;
  Lcoords[1] = 0.0;
  phys[0] = tri0->GetCoord(0, Lcoords);
  phys[1] = tri0->GetCoord(1, Lcoords);
  ref[0] = Lcoords[0];
  ref[1] = Lcoords[1];
  lambda_map_to_phys_tri(ref, test_phys);
  lambda_map_to_ref_tri(phys,  test_out);
  nprint("(", Lcoords[0], "," ,Lcoords[1], ") -> (",
              phys[0], ",", phys[1], "|" , test_phys[0], test_phys[1],  ") => (", 
              test_out[0], ",", test_out[1], ")");
  

  Lcoords[0] = -1.0;
  Lcoords[1] = 0.0;
  phys[0] = tri0->GetCoord(0, Lcoords);
  phys[1] = tri0->GetCoord(1, Lcoords);
  ref[0] = Lcoords[0];
  ref[1] = Lcoords[1];
  lambda_map_to_phys_tri(ref, test_phys);
  lambda_map_to_ref_tri(phys,  test_out);
  nprint("(", Lcoords[0], "," ,Lcoords[1], ") -> (",
              phys[0], ",", phys[1], "|" , test_phys[0], test_phys[1],  ") => (", 
              test_out[0], ",", test_out[1], ")");

  Lcoords[0] = -0.5;
  Lcoords[1] = 0.0;
  phys[0] = tri0->GetCoord(0, Lcoords);
  phys[1] = tri0->GetCoord(1, Lcoords);
  ref[0] = Lcoords[0];
  ref[1] = Lcoords[1];
  lambda_map_to_phys_tri(ref, test_phys);
  lambda_map_to_ref_tri(phys,  test_out);
  nprint("(", Lcoords[0], "," ,Lcoords[1], ") -> (",
              phys[0], ",", phys[1], "|" , test_phys[0], test_phys[1],  ") => (", 
              test_out[0], ",", test_out[1], ")");


  Lcoords[0] = 0.0;
  Lcoords[1] = -0.5;
  phys[0] = tri0->GetCoord(0, Lcoords);
  phys[1] = tri0->GetCoord(1, Lcoords);
  ref[0] = Lcoords[0];
  ref[1] = Lcoords[1];
  lambda_map_to_phys_tri(ref, test_phys);
  lambda_map_to_ref_tri(phys,  test_out);
  nprint("(", Lcoords[0], "," ,Lcoords[1], ") -> (",
              phys[0], ",", phys[1], "|" , test_phys[0], test_phys[1],  ") => (", 
              test_out[0], ",", test_out[1], ")");



  nprint("============================================");




  nprint("............................................");

  auto foo2 = tri0->GetCoeffs(0);
  nprint("foo2.size()", foo2.size());
  nprint("foo2:", foo2[0], foo2[1], foo2[2]);
  auto foo3 = tri0->GetCoeffs(1);
  nprint("foo3.size()", foo3.size());
  nprint("foo3:", foo3[0], foo3[1], foo3[2]);

   auto lambda_map_to_phys_trif = [&] (double * in, double * out) {
    //const double x0 = in[0];
    //const double x1 = in[1];
    //const double c0 = (1.0 - x0) * (1.0 - x1) * 0.25;
    //const double c1 = (1.0 - x0) * (1.0 + x1) * 0.25;
    //const double c2 = (1.0 + x0) * (1.0 - x1) * 0.25;
    //
    //out[0] = foo2[0] * c0 + foo2[1] * c1 + foo2[2] * c2;
    //out[1] = foo3[0] * c0 + foo3[1] * c1 + foo3[2] * c2;
    
    double M[4];
    M[0] = 0.5 * (foo2[2] - foo2[0]);
    M[1] = 0.5 * (foo2[1] - foo2[0]);
    M[2] = 0.5 * (foo3[2] - foo3[0]);
    M[3] = 0.5 * (foo3[1] - foo3[0]);
  
    double s[2];
    s[0] = in[0] + 1.0;
    s[1] = in[1] + 1.0;
    
    double mv[2];

    mv[0] = M[0] * s[0] + M[1] * s[1];
    mv[1] = M[2] * s[0] + M[3] * s[1];
    
    out[0] = mv[0] + foo2[0];
    out[1] = mv[1] + foo3[0];
  }; 
  
  
  test_in[0] = -1.0;
  test_in[1] = -1.0;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);
  Lcoords[0] = -1.0;
  Lcoords[1] = -1.0;
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));


  test_in[0] = 1.0;
  test_in[1] = -1.0;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);

  lambda_n_map(test_in, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));

  test_in[0] = -1.0;
  test_in[1] = 1.0;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);
  lambda_n_map(test_in, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));


  nprint("============================================");
  test_in[0] = -0.5;
  test_in[1] = -1.0;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);
  lambda_n_map(test_in, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));
 

  test_in[0] = -0.25;
  test_in[1] = -1.0;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);
  lambda_n_map(test_in, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));

  nprint("============================================");

  test_in[0] = -1.0;
  test_in[1] = -0.5;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);
  lambda_n_map(test_in, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));
 

  test_in[0] = -1.0;
  test_in[1] = -0.25;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);
  lambda_n_map(test_in, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));

  nprint("============================================");


  test_in[0] = -0.5;
  test_in[1] = -0.5;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);
  lambda_n_map(test_in, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));


  test_in[0] = -0.9;
  test_in[1] = -0.9;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);
  lambda_n_map(test_in, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));


  test_in[0] = -0.9;
  test_in[1] = 0.7;
  lambda_map_to_phys_trif(test_in, test_out);
  nprint("test_in:", test_in[0], test_in[1], "test_out:", test_out[0], test_out[1]);
  lambda_n_map(test_in, eta);
  Lcoords[0] = eta[0];
  Lcoords[1] = eta[1];
  nprint("n", Lcoords[0], Lcoords[1], "->", tri0->GetCoord(0, Lcoords), tri0->GetCoord(1, Lcoords));


  nprint("============================================");



  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapperT>(sycl_target, mesh, tol);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  const double extent[2] = {1.0, 1.0};
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

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

      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain->mesh, A->position_dat, A->cell_id_dat,
      A->mpi_rank_dat);

  auto lambda_check_owning_cell = [&] {
    auto point = std::make_shared<PointGeom>(ndim, -1, 0.0, 0.0, 0.0);
    Array<OneD, NekDouble> global_coord(3);
    Array<OneD, NekDouble> local_coord(3);
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A->position_dat->cell_dat.get_cell(cellx);
      auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
      auto reference_positions =
          (*A)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]->cell_dat.get_cell(cellx);

      for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {

        const int cell_neso = (*cell_ids)[0][rowx];
        ASSERT_EQ(cell_neso, cellx);
        const int cell_nektar = cell_id_translation.map_to_nektar[cell_neso];

        global_coord[0] = (*positions)[0][rowx];
        global_coord[1] = (*positions)[1][rowx];

        NekDouble dist;
        auto geom = graph->GetGeometry2D(cell_nektar);
        auto is_contained =
            geom->ContainsPoint(global_coord, local_coord, tol, dist);

        ASSERT_TRUE(is_contained);

        // check the local coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          ASSERT_TRUE(ABS(local_coord[dimx] -
                          (*reference_positions)[dimx][rowx]) <= tol);
        }
      }
    }
  };

  pbc.execute();
  mesh_hierarchy_global_map.execute();
  A->hybrid_move();
  cell_id_translation.execute();
  A->cell_move();
  lambda_check_owning_cell();

  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
}
