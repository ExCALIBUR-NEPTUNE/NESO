#ifndef __READERS_H__
#define __READERS_H__

#include "AtomicDataReader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class CSVAtomicDataReader : public AtomicDataReader {
public:
  CSVAtomicDataReader(std::string filepath) : AtomicDataReader(filepath) { read(); };
  virtual void read() final {
    std::fstream file(m_filepath, std::ios::in);
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
      while (getline(file, line)) {
        row.clear();
        std::stringstream str(line);
        while (getline(str, word, ','))
          row.push_back(word);
        content.push_back(row);
      }
    } else {
      std::cerr << "CSVAtomicDataReader: " << m_filepath << " not found" << std::endl;
      return;
    }
    int nrows = content.size();
    m_data[m_T_idx] = std::vector<double>(nrows);
    m_data[m_rate_idx] = std::vector<double>(nrows);
    for (int i = 0; i < nrows; i++) {
      try {
        m_data[m_T_idx][i] = std::stod(content[i][m_T_idx]);
        m_data[m_rate_idx][i] = std::stod(content[i][m_rate_idx]);
      } catch (const std::invalid_argument &ia_ex) {
        std::cerr << "CSVAtomicDataReader: Invalid argument converting value on line number ["
                  << i + 2 << "] to double in file " << m_filepath << std::endl;
        throw;
      } catch (const std::out_of_range &oor_ex) {
        std::cerr << "CSVAtomicDataReader: Out-of-double-range argument converting value on line number ["
                  << i + 2 << "] to double in file " << m_filepath << std::endl;
        throw;
      }
    }
  }
};
#endif
