#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/Timer.h>
#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>
#include <memory>


#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include "nektar_interface/utility_mesh_plotting.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <neso_particles.hpp>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::LibUtilities;
using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;
using namespace NESO::CompositeInteraction;

void mesh_plotting_inner(int argc, char **argv,
                         LibUtilities::SessionReaderSharedPtr session,
                         SpatialDomains::MeshGraphSharedPtr graph) {

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto comm = mesh->get_comm();
  auto sycl_target = std::make_shared<SYCLTarget>(0, comm);
  write_vtk_cells_owned("mesh_owned", mesh);
  write_vtk_cells_halo("mesh_halo_default", mesh);
  write_vtk_mesh_hierarchy_cells_owned("mesh_mh", mesh);
  write_vtk_mesh_hierarchy_cells_fine("mesh_mh_fine.vtk", mesh);
  write_vtk_mesh_hierarchy_cells_coarse("mesh_mh_coarse.vtk", mesh);

  sycl_target->free();
  mesh->free();
}

void mesh_plotting_inner_halos(int argc, char **argv,
                               LibUtilities::SessionReaderSharedPtr session,
                               SpatialDomains::MeshGraphSharedPtr graph,
                               const int halo_stencil_width,
                               const int halo_stencil_pbc) {

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  extend_halos_fixed_offset(halo_stencil_width, mesh, (bool)halo_stencil_pbc);
  auto comm = mesh->get_comm();
  auto sycl_target = std::make_shared<SYCLTarget>(0, comm);
  write_vtk_cells_halo("mesh_halo_width_" + std::to_string(halo_stencil_width),
                       mesh);

  sycl_target->free();
  mesh->free();
}
