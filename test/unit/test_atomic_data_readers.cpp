#include "csv_atomic_data_reader.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace NESO;

TEST(AtomicDataReadersTest, CSV) {

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir = source_dir / "../test_resources";
  std::filesystem::path csv_test_file =
      test_resources_dir / "charge_exchange_h0_h1.csv";

  CSVAtomicDataReader atomic_data =
      CSVAtomicDataReader(std::string(csv_test_file));
  std::vector<std::vector<double>> all_data = atomic_data.get_data();
  std::vector<double> rates = atomic_data.get_rates();
  std::vector<double> temps = atomic_data.get_temps();

  ASSERT_NEAR(all_data[0][0], 1.0067508210214045, 1e-16);
  ASSERT_NEAR(all_data[1][0], 9.762958524529584e-9, 1e-16);
  ASSERT_NEAR(all_data[0][83], 101361.54790248517, 1e-16);
  ASSERT_NEAR(all_data[1][83], 2.1008833642746626e-8, 1e-16);

  ASSERT_NEAR(temps[0], 1.0067508210214045, 1e-16);
  ASSERT_NEAR(rates[0], 9.762958524529584e-9, 1e-16);
  ASSERT_NEAR(temps[83], 101361.54790248517, 1e-16);
  ASSERT_NEAR(rates[83], 2.1008833642746626e-8, 1e-16);

  // Save current stderr buffer
  std::streambuf *saved_stderr = std::cerr.rdbuf();

  // Redirect stderr to a stringstream buffer
  std::stringstream ss;
  std::cerr.rdbuf(ss.rdbuf());

  csv_test_file =
      test_resources_dir / "bad_invalid_argument_charge_exchange_h0_h1.csv";
  ASSERT_THROW(CSVAtomicDataReader(std::string(csv_test_file)),
               std::invalid_argument);

  csv_test_file =
      test_resources_dir / "bad_out_of_double_range_charge_exchange_h0_h1.csv";
  ASSERT_THROW(CSVAtomicDataReader(std::string(csv_test_file)),
               std::out_of_range);

  csv_test_file = test_resources_dir / "non_existent_file.csv";
  ASSERT_THROW(CSVAtomicDataReader(std::string(csv_test_file)),
               std::ios_base::failure);

  // Restore stderr buffer
  std::cerr.rdbuf(saved_stderr);
}
