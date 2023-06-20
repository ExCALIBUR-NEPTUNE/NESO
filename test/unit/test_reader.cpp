#include "../include/readers.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

TEST(ReadersTest, CSV) {

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir = source_dir / "../test_resources";
  std::filesystem::path csv_test_file =
      test_resources_dir / "charge_exchange_h0_h1.csv";

  std::vector<std::vector<double>> all_data =
      CSVReader(std::string(csv_test_file)).GetData();
  std::vector<double> rates = CSVReader(std::string(csv_test_file)).GetRates();
  std::vector<double> temps = CSVReader(std::string(csv_test_file)).GetTemps();

  ASSERT_NEAR(all_data[0][0], 1.00, 1.01);
  ASSERT_NEAR(all_data[1][0], 9.76e-9, 9.767e-9);
  ASSERT_NEAR(all_data[0][83], 101361.54, 101361.55);
  ASSERT_NEAR(all_data[1][83], 2.10e-8, 2.11e-8);

  ASSERT_NEAR(temps[0], 1.00, 1.01);
  ASSERT_NEAR(rates[0], 9.76e-9, 9.767e-9);
  ASSERT_NEAR(temps[83], 101361.54, 101361.55);
  ASSERT_NEAR(rates[83], 2.10e-8, 2.11e-8);
}
