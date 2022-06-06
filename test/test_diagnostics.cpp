#include "diagnostics.hpp"
#include <cmath>
#include <gtest/gtest.h>

TEST(DiagnosticsTest, Diagnostics) {

  Diagnostics diagnostics;
  EXPECT_EQ(diagnostics.total_energy.size(), 0);
  EXPECT_EQ(diagnostics.particle_energy.size(), 0);
  EXPECT_EQ(diagnostics.field_energy.size(), 0);
}

/*
 * Test that the size of the energy vectors
 * increment in size by one every time
 * compute_total_energy is called.
 */
TEST(DiagnosticsTest, SizeIncrement) {

  auto asyncHandler = [&](sycl::exception_list exceptionList) {};
  auto Q = sycl::queue{sycl::default_selector{}, asyncHandler};
  Mesh mesh(10);
  Species electrons(mesh, true, 1.0, 1.0, 1.0, 100);
  std::vector<Species> species_list;
  species_list.push_back(electrons);
  Plasma plasma(species_list);
  Diagnostics diagnostics;
  FFT fft(Q, mesh.nintervals);

  mesh.set_initial_field(Q, mesh, plasma, fft);
  diagnostics.compute_total_energy(Q, mesh, plasma);

  EXPECT_EQ(diagnostics.total_energy.size(), 1);
  EXPECT_EQ(diagnostics.particle_energy.size(), 1);
  EXPECT_EQ(diagnostics.field_energy.size(), 1);

  diagnostics.compute_total_energy(Q, mesh, plasma);

  EXPECT_EQ(diagnostics.total_energy.size(), 2);
  EXPECT_EQ(diagnostics.particle_energy.size(), 2);
  EXPECT_EQ(diagnostics.field_energy.size(), 2);
}

/*
 * Test that the total energy is the sum of
 * the particle energy and the field energy
 */
TEST(DiagnosticsTest, TotalIsSum) {

  auto asyncHandler = [&](sycl::exception_list exceptionList) {};
  auto Q = sycl::queue{sycl::default_selector{}, asyncHandler};
  Mesh mesh(10);
  Species electrons(mesh, true, 1.0, 1.0, 1.0, 100);
  std::vector<Species> species_list;
  species_list.push_back(electrons);
  Plasma plasma(species_list);
  Diagnostics diagnostics;
  FFT fft(Q, mesh.nintervals);

  mesh.set_initial_field(Q, mesh, plasma, fft);
  diagnostics.compute_total_energy(Q, mesh, plasma);

  diagnostics.compute_total_energy(Q, mesh, plasma);

  for (int i = 0; i < diagnostics.total_energy.size(); i++) {
    ASSERT_EQ(diagnostics.total_energy.at(i),
              diagnostics.particle_energy.at(i) +
                  diagnostics.field_energy.at(i));
  }
}

/*
 * Test that the energy of a species is proportional to its mass
 */
TEST(DiagnosticsTest, ProportionalToMass) {

  auto asyncHandler = [&](sycl::exception_list exceptionList) {};
  auto Q = sycl::queue{sycl::default_selector{}, asyncHandler};
  Mesh mesh(10);
  Species electrons(mesh, true, 1.0, 1.0, 1.0, 100);
  std::vector<Species> species_list;
  species_list.push_back(electrons);
  Plasma plasma(species_list);
  Diagnostics diagnostics;
  FFT fft(Q, mesh.nintervals);

  mesh.set_initial_field(Q, mesh, plasma, fft);
  diagnostics.compute_total_energy(Q, mesh, plasma);

  Species electrons2(mesh, true, 1.0, 1.0, 2.0, 100);
  std::vector<Species> species_list2;
  species_list2.push_back(electrons2);
  Plasma plasma2(species_list2);
  Diagnostics diagnostics2;

  mesh.set_initial_field(Q, mesh, plasma2, fft);
  diagnostics2.compute_total_energy(Q, mesh, plasma2);

  double ratio =
      diagnostics2.total_energy.at(0) / diagnostics.total_energy.at(0);

  // Since initial fields are random, these can be
  // surprisingly far from each other
  ASSERT_NEAR(2.0, ratio, 0.01);
}
