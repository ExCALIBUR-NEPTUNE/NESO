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

  auto prototype_function = std::make_shared<DisContField>(session, graph, "u");

  auto composite_function_context = std::make_shared<CompositeFunctionContext>(
      sycl_target, graph, prototype_function, boundary_groups);

  auto func0 = composite_function_context->create_function(0);
  auto func1 = composite_function_context->create_function(1);

  ASSERT_EQ(func0->exp_lists.size(), 3);
  ASSERT_EQ(func1->exp_lists.size(), 3);

  auto lambda_get_geoms = [&](auto &func) -> std::vector<INT> {
    std::vector<INT> tmp_geoms;

    for (auto &exp_list : func->exp_lists) {
      if (exp_list != nullptr) {
        const auto exp_list_size = exp_list->GetExpSize();
        for (int ex = 0; ex < exp_list_size; ex++) {
          auto geom = exp_list->GetExp(ex)->GetGeom();
          const int geom_id = geom->GetGlobalID();
          tmp_geoms.push_back(geom_id);
        }
      }
    }

    return tmp_geoms;
  };

  ASSERT_EQ(lambda_get_geoms(func0),
            composite_function_context->get_owned_geoms(0));
  ASSERT_EQ(lambda_get_geoms(func1),
            composite_function_context->get_owned_geoms(1));

  sycl_target->free();
}
