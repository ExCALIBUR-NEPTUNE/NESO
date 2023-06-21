#ifndef __READERS_H__
#define __READERS_H__

#include "AtomicDataReader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class CSVAtomicReader : public AtomicDataReader {
public:
  CSVReader(std::string filepath) : AtomicDataReader(filepath) { Read(); };
  virtual void Read() final {
    std::fstream file(m_filepath, std::ios::in);
    if (file.is_open()) {
      std::string line;
      getline(file, line);
      if (line == "DEFAULT") {
        T_idx = 0;
        rate_idx = 1;
      }
      std::vector<std::vector<std::string>> content;
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
      std::cerr << "CSVReader: " << m_filepath << " not found" << std::endl;
      return;
    }
    int nrows = content.size();
    m_data[T_idx] = std::vector<double>(nrows);
    m_data[rate_idx] = std::vector<double>(nrows);
    for (int i = 0; i < nrows; i++) {
      try {
        m_data[0][i] = std::stod(content[i][0]);
        m_data[1][i] = std::stod(content[i][1]);
      } catch (const std::invalid_argument &ia_ex) {
        // Error handling
        std::cerr << "CSVReader: Invalid argument converting value on line number ["
                  << i + 2 << "] to double in file " << m_filepath << std::endl;
        throw;
      } catch (const std::out_of_range &oor_ex) {
        // Error handling
        std::cerr << "CSVReader: Out-of-double-range argument converting value on line number ["
                  << i + 2 << "] to double in file " << m_filepath << std::endl;
        throw;
      }
    }
  }
};
#endif
