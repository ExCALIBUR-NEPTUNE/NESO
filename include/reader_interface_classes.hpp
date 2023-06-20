#ifndef __READER_H__
#define __READER_H__

#include <string>
#include <vector>

class Reader {
public:
  Reader(std::string filepath) : m_data(2), m_filepath(filepath) {}
  Reader() = delete;
  std::vector<std::vector<double>> GetData() { return m_data; }
  std::vector<double> GetRates() { return m_data[rate_idx]; }
  std::vector<double> GetTemps() { return m_data[T_idx]; }

protected:
  int T_idx;
  int rate_idx;

  std::string m_filepath;

  virtual void Read() = 0;
  std::vector<std::vector<double>> m_data;
};
#endif
