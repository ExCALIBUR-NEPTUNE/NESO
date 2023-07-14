#include "nektar_interface/parameter_store.hpp"
#include <gtest/gtest.h>
#include <neso_particles.hpp>

using namespace NESO;
using namespace NESO::Particles;

TEST(ParameterStore, GetSet) {

  ParameterStore ps{};
  ps.set<INT>("a", 42);
  ps.set<REAL>("ar", 3.1415);

  ASSERT_EQ(ps.get<INT>("a"), 42);
  ASSERT_EQ(ps.get<REAL>("ar"), 3.1415);
  ASSERT_EQ(ps.get<INT>("b", 36), 36);
  ASSERT_EQ(ps.get<REAL>("b", 3.6), 3.6);
}
