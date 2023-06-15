#include <string>
#include <vector>
#include "../include/readers.hpp"

TEST(ReadersTest, CSV) {

   TabulatedDataReaderCSV test;
   
   std::vector<std::vector<double>> a=test.ReadData("../test_resources/charge_exchange_h0_h1.csv");
   ASSERT_NEAR(a[0][0],1.00,1.01)
   ASSERT_NEAR(a[0][1],9.76e-9,9.767e-9)
   ASSERT_NEAR(a[83][0],101361.54,101361.55)
   ASSERT_NEAR(a[83][1],2.10e-8,2.11e-8)
}	
