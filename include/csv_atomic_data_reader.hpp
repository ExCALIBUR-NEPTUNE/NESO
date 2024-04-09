#ifndef __CSV_ATOMIC_DATA_READER_H__
#define __CSV_ATOMIC_DATA_READER_H__

#include "atomic_data_reader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace NESO {

/**
 *  Class used to read atomic data from a 2 column csv file, returning
 *  that data to the 2d vector m_data. m_data is used by functions
 *  get_data, get_temps and get_data derived from class AtomicDataReader
 *  the read in data to the user.
 */

class CSVAtomicDataReader : public AtomicDataReader {
public:
  CSVAtomicDataReader(std::string filepath) : AtomicDataReader(filepath) {
    read();
  };

protected:
  virtual void read() final {
    // Open the file; enable exceptions on error
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      file.open(m_filepath);
    } catch (std::ifstream::failure e) {
      std::cerr << "CSVAtomicDataReader: Failed to open file " << m_filepath
                << std::endl;
      throw;
    }

    std::vector<std::vector<std::string>> content;
    if (file.is_open()) {
      std::string line;
      getline(file, line);
      if (line == "DEFAULT") {
        m_T_idx = 0;
        m_rate_idx = 1;
      }
      std::vector<std::string> row;
      std::string word;
      // Read the next line, stopping before EOF to avoid an exception
      while (file.peek() != EOF && getline(file, line)) {
        row.clear();
        std::stringstream str(line);
        while (getline(str, word, ','))
          row.push_back(word);
        content.push_back(row);
      }
    }

    int nrows = content.size();
    m_data[m_T_idx] = std::vector<double>(nrows);
    m_data[m_rate_idx] = std::vector<double>(nrows);
    for (int i = 0; i < nrows; i++) {
      try {
        m_data[m_T_idx][i] = std::stod(content[i][m_T_idx]);
        m_data[m_rate_idx][i] = std::stod(content[i][m_rate_idx]);
      } catch (const std::invalid_argument &ia_ex) {
        std::cerr << "CSVAtomicDataReader: Invalid argument converting value "
                     "on line number ["
                  << i + 2 << "] to double in file " << m_filepath << std::endl;
        throw;
      } catch (const std::out_of_range &oor_ex) {
        std::cerr << "CSVAtomicDataReader: Out-of-double-range argument "
                     "converting value on line number ["
                  << i + 2 << "] to double in file " << m_filepath << std::endl;
        throw;
      }
    }
  }
};

} // namespace NESO
#endif
