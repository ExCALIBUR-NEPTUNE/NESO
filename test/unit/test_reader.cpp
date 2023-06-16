#include "../include/readers.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

TEST(ReadersTest, CSV) {

  TabulatedDataReaderCSV test;
  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir = source_dir / "../test_resources";
  std::filesystem::path csv_test_file =
      test_resources_dir / "charge_exchange_h0_h1.csv";

  std::vector<std::vector<double>> a =
      test.ReadData(std::string(csv_test_file));

  ASSERT_NEAR(a[0][0], 1.00, 1.01);
  ASSERT_NEAR(a[0][1], 9.76e-9, 9.767e-9);
  ASSERT_NEAR(a[83][0], 101361.54, 101361.55);
  ASSERT_NEAR(a[83][1], 2.10e-8, 2.11e-8);
}
