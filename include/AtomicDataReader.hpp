#ifndef __READER_H__
#define __READER_H__

#include <string>
#include <vector>

class AtomicDataReader {
public:
  AtomicDataReader(std::string filepath) : m_data(2), m_filepath(filepath) {}
  AtomicDataReader() = delete;
  std::vector<std::vector<double>> get_data() { return m_data; }
  std::vector<double> get_rates() { return m_data[m_rate_idx]; }
  std::vector<double> get_temps() { return m_data[m_T_idx]; }

protected:
  int m_T_idx;
  int m_rate_idx;

  std::string m_filepath;

  virtual void read() = 0;
  std::vector<std::vector<double>> m_data;
};
#endif
