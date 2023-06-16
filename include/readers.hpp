#ifndef __READERS_H__
#define __READERS_H__

#include "reader_interface_classes.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class TabulatedDataReaderCSV : public ITabulatedDataReader {
public:
  // read both columns of a 2 column csv file and output
  // first column to 1D vector
  std::vector<double> GridData(std::string filename) {
    std::vector<std::vector<std::string>> content_file;

    std::vector<double> temp;
    for (int i = 0; i < content_file.size(); i++) {
      temp.push_back(std::stod(content_file[i][0]));
    }
    return (temp);
  }

  // read both columns of a 2 column csv file and output
  // second column to 1D vector
  std::vector<double> RatesData(std::string filename) {
    std::vector<std::vector<std::string>> content_file;

    std::vector<double> rates;
    content_file = CSVtostring(filename);
    for (int i = 0; i < content_file.size(); i++) {
      rates.push_back(std::stod(content_file[i][1]));
    }
    return (rates);
  }

  // read both columns of a 2 column csv file to a 2d vector
  std::vector<std::vector<double>> ReadData(std::string filename) {
    std::vector<std::vector<std::string>> content_file;

    std::vector<std::vector<double>> temp_and_rates;
    content_file = CSVtostring(filename);
    for (int i = 0; i < content_file.size(); i++) {
      temp_and_rates.push_back(
          {std::stod(content_file[i][0]), std::stod(content_file[i][1])});
    }
    return (temp_and_rates);
  }

private:
  // Reads 2 column csv file to a 2D string vector to be converted and used by
  // ReadData, GridData and RatesData
  std::vector<std::vector<std::string>> CSVtostring(std::string filename) {
    std::vector<std::vector<std::string>> content;
    std::vector<std::string> row;
    std::string line, word;
    std::fstream file(filename, std::ios::in);
    if (file.is_open()) {
      while (getline(file, line)) {
        row.clear();
        std::stringstream str(line);
        while (getline(str, word, ','))
          row.push_back(word);
        content.push_back(row);
      }
    }
    return (content);
  }
};

#endif
