#ifndef __READER_H__
#define __READER_H__

#include <string>
#include <vector>

/**
 *  Base class that defines functions and variables needed to read in atomic
 * data. You can return a 2D vector which contains all the data in get_data, or
 * a 1D vector which contains only the information you are interested in, e.g.
 * temperatures, rates via the function calls get_temps and get get_rates
 * respectively.
 *
 * Derived classes must override read() in order to populate m_data and set the
 * *_idx member variables.
 */
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

  const std::string m_filepath;

  virtual void read() = 0;
  std::vector<std::vector<double>> m_data;
};
#endif
