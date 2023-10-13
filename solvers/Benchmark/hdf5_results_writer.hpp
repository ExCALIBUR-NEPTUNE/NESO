#ifndef __HDF5_RESULTS_WRITER_H_
#define __HDF5_RESULTS_WRITER_H_

#include <string>
#include <vector>

#include <neso_particles.hpp>
using namespace NESO::Particles;

class HDF5ResultsWriter {
protected:
public:
  hid_t file;
  HDF5ResultsWriter(const std::string filename) {
    file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  };

  inline void add_array(const std::string name, const int nrow, const int ncol,
                        const std::vector<double> &data) {
    hsize_t dims[2] = {static_cast<hsize_t>(nrow), static_cast<hsize_t>(ncol)};
    hid_t dataspace = H5Screate_simple(2, dims, nullptr);
    hid_t dataset = H5Dcreate2(file, name.c_str(), H5T_NATIVE_DOUBLE, dataspace,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5CHK(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   data.data()));

    H5CHK(H5Dclose(dataset));
    H5CHK(H5Sclose(dataspace));
  }

  inline void add_parameter(const std::string key, const int value) {
    hid_t dataspace = H5Screate(H5S_SCALAR);
    hid_t attribute = H5Acreate(file, key.c_str(), H5T_NATIVE_INT, dataspace,
                                H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Awrite(attribute, H5T_NATIVE_INT, &value);
    H5Aclose(attribute);
    H5Sclose(dataspace);
  }

  ~HDF5ResultsWriter() { H5CHK(H5Fclose(file)); };
};

#endif
