#ifndef __READER_H__
#define __READER_H__

#include <string>
#include <vector>

class ITabulatedDataReader {
public:
  virtual std::vector<double> RatesData(std::string filename) = 0;
  virtual std::vector<double> GridData(std::string filename) = 0;
  virtual std::vector<std::vector<double>> ReadData(std::string filename) = 0;

protected:
  virtual std::vector<std::vector<std::string>>
  CSVtostring(std::string filename) = 0;
};
#endif
