#include <gtest/gtest.h>
#include "../src/diagnostics.hpp"
#include <cmath>

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

  Mesh mesh(10);
  Plasma plasma(100);
  Diagnostics diagnostics;
  FFT fft(mesh.nintervals);

  mesh.set_initial_field(&mesh,&plasma,&fft);
  diagnostics.compute_total_energy(&mesh,&plasma);

  EXPECT_EQ(diagnostics.total_energy.size(), 1);
  EXPECT_EQ(diagnostics.particle_energy.size(), 1);
  EXPECT_EQ(diagnostics.field_energy.size(), 1);

  diagnostics.compute_total_energy(&mesh,&plasma);

  EXPECT_EQ(diagnostics.total_energy.size(), 2);
  EXPECT_EQ(diagnostics.particle_energy.size(), 2);
  EXPECT_EQ(diagnostics.field_energy.size(), 2);

}

/*
 * Test that the total energy is the sum of
 * the particle energy and the field energy
 */
TEST(DiagnosticsTest, TotalIsSum) {

  Mesh mesh(10);
  Plasma plasma(100);
  Diagnostics diagnostics;
  FFT fft(mesh.nintervals);

  mesh.set_initial_field(&mesh,&plasma,&fft);
  diagnostics.compute_total_energy(&mesh,&plasma);

  diagnostics.compute_total_energy(&mesh,&plasma);

  for(int i = 0; i < diagnostics.total_energy.size(); i++){
		ASSERT_EQ( diagnostics.total_energy.at(i), diagnostics.particle_energy.at(i) + diagnostics.field_energy.at(i) );
  }
}
