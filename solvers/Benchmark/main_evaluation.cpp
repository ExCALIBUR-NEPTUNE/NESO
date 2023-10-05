#include "main_evaluation.hpp"

int main_evaluation(int argc, char *argv[], LibUtilities::SessionReaderSharedPtr session) {
  int err = 0;

  // Create MeshGraph.
  auto graph = SpatialDomains::MeshGraph::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);







  return err;
}
