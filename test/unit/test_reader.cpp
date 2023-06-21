#include "CSVAtomicReader.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

TEST(AtomicDataReadersTest, CSV) {

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir = source_dir / "../test_resources";
  std::filesystem::path csv_test_file =
      test_resources_dir / "charge_exchange_h0_h1.csv";

  std::vector<std::vector<double>> all_data = CSVAtomicReader(std::string(csv_test_file)).GetData();
  std::vector<double> rates = CSVAtomicReader(std::string(csv_test_file)).GetRates();
  std::vector<double> temps = CSVAtomicReader(std::string(csv_test_file)).GetTemps();

  ASSERT_NEAR(all_data[0][0], 1.0067508210214045, 1e-16);
  ASSERT_NEAR(all_data[1][0], 9.762958524529584e-9, 1e-16);
  ASSERT_NEAR(all_data[0][83], 101361.54790248517, 1e-16);
  ASSERT_NEAR(all_data[1][83], 2.1008833642746626e-8, 1e-16);

  ASSERT_NEAR(temps[0], 1.0067508210214045, 1e-16);
  ASSERT_NEAR(rates[0], 9.762958524529584e-9, 1e-16);
  ASSERT_NEAR(temps[83], 101361.54790248517, 1e-16);
  ASSERT_NEAR(rates[83], 2.1008833642746626e-8, 1e-16);
}
