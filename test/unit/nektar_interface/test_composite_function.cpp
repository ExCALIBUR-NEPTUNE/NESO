#include <SpatialDomains/MeshGraphIO.h>

#include "../../unit/nektar_interface/test_helper_utilities.hpp"

using namespace CompositeInteraction;


TEST(CompositeInteraction, SurfaceFunction3DInit) {

  const std::string filename_conditions =
      "reference_all_types_cube/conditions.xml";
  const std::string filename_mesh =
      "reference_all_types_cube/linear_non_regular_0.5.xml";
  const int ndim = 3;

  TestUtilities::TestResourceSession resources_session(filename_mesh,
                                                       filename_conditions);
  auto session = resources_session.session;
  auto graph = SpatialDomains::MeshGraphIO::Read(session);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);


  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {100, 200, 300};
  boundary_groups[1] = {400, 500, 600};







  


  sycl_target->free();
}

