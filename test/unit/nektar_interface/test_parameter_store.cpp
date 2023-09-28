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

TEST(ParameterStore, InitInt) {
  ParameterStore ps(std::map<std::string, INT>({{"a", 42}}));
  ASSERT_EQ(ps.get<INT>("a"), 42);
  ASSERT_EQ(ps.get<INT>("b", 36), 36);
}

TEST(ParameterStore, InitREAL) {
  ParameterStore ps(std::map<std::string, REAL>({{"a", 4.123}}));
  ps.set<REAL>("ar", 0.123);
  ASSERT_EQ(ps.get<REAL>("ar"), 0.123);
  ASSERT_EQ(ps.get<REAL>("a"), 4.123);
  ASSERT_EQ(ps.get<REAL>("b", 3.6), 3.6);
}

TEST(ParameterStore, InitREALINT) {
  ParameterStore ps(std::map<std::string, INT>({{"a", 42}}),
                    std::map<std::string, REAL>({{"ar", 3.1415}}));
  ASSERT_EQ(ps.get<INT>("a"), 42);
  ASSERT_EQ(ps.get<REAL>("ar"), 3.1415);
  ASSERT_EQ(ps.get<INT>("b", 36), 36);
  ASSERT_EQ(ps.get<REAL>("b", 3.6), 3.6);
}

TEST(ParameterStore, InitREALINTShared) {
  auto ps = std::make_shared<ParameterStore>(
      std::map<std::string, INT>({{"a", 42}}),
      std::map<std::string, REAL>({{"ar", 3.1415}}));
  ASSERT_EQ(ps->get<INT>("a"), 42);
  ASSERT_EQ(ps->get<REAL>("ar"), 3.1415);
  ASSERT_EQ(ps->get<INT>("b", 36), 36);
  ASSERT_EQ(ps->get<REAL>("b", 3.6), 3.6);
}
