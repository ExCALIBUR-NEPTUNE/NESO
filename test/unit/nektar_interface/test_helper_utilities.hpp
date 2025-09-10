#ifndef _TEST_UNIT_NEKTAR_INTERFACE_TEST_HELPER_UTILTIES_HPP_
#define _TEST_UNIT_NEKTAR_INTERFACE_TEST_HELPER_UTILTIES_HPP_

#include "nektar_interface/basis_reference.hpp"
#include "nektar_interface/composite_interaction/composite_interaction.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include "nektar_interface/utility_mesh.hpp"
#include "nektar_interface/utility_mesh_plotting.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/Foundations/GaussPoints.h>
#include <LibUtilities/Foundations/PolyEPoints.h>
#include <SolverUtils/Driver.h>
#include <SpatialDomains/MeshGraphIO.h>
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

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::LibUtilities;
using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

namespace NESO::TestUtilities {

/**
 * Type to aid loading test resources into Nektar++ sessions.
 */
class TestResourceSession {
protected:
  char *argv[3]{nullptr, nullptr, nullptr};

public:
  /// The session created from test resources.
  LibUtilities::SessionReaderSharedPtr session;

  ~TestResourceSession() {
    for (int ix = 0; ix < 3; ix++) {
      auto ptr = this->argv[ix];
      if (ptr != nullptr) {
        delete[] ptr;
        this->argv[ix] = nullptr;
      }
    }
  }

  /**
   * Create a Nektar++ session from a conditions file and mesh file in the
   * test_resources directory.
   *
   * @param filename_mesh Path relative to the test_resources directory for the
   * mesh.
   * @param filename_conditions Path relative to the test_resources directory
   * for the conditions.
   */
  TestResourceSession(const std::string filename_mesh,
                      const std::string filename_conditions) {

    copy_to_cstring(std::string("neso_nektar_test"), &this->argv[0]);

    std::filesystem::path source_file = __FILE__;
    std::filesystem::path source_dir = source_file.parent_path();
    std::filesystem::path test_resources_dir =
        source_dir / "../../test_resources";

    std::filesystem::path conditions_file =
        test_resources_dir / filename_conditions;
    copy_to_cstring(std::string(conditions_file), &this->argv[1]);
    std::filesystem::path mesh_file = test_resources_dir / filename_mesh;
    copy_to_cstring(std::string(mesh_file), &this->argv[2]);

    // Create session reader.
    session = LibUtilities::SessionReader::CreateInstance(3, this->argv);
  }

  /**
   * Create a Nektar++ session from a conditions file and mesh file in the
   * test_resources directory.
   *
   * @param filename_mesh Path relative to the test_resources directory for the
   * mesh.
   */
  TestResourceSession(const std::string filename_mesh) {

    copy_to_cstring(std::string("neso_nektar_test"), &this->argv[0]);

    std::filesystem::path source_file = __FILE__;
    std::filesystem::path source_dir = source_file.parent_path();
    std::filesystem::path test_resources_dir =
        source_dir / "../../test_resources";

    std::filesystem::path mesh_file = test_resources_dir / filename_mesh;
    copy_to_cstring(std::string(mesh_file), &this->argv[1]);

    // Create session reader.
    session = LibUtilities::SessionReader::CreateInstance(2, this->argv);
  }
};

} // namespace NESO::TestUtilities

#endif
