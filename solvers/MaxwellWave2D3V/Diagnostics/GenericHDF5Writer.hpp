#ifndef __GENERIC_HDF5_WRITER_H_
#define __GENERIC_HDF5_WRITER_H_

#include <hdf5.h>
#include <string>

class GenericHDF5Writer {
private:
  std::string filename;
  int step;
  hid_t file;
  hid_t group_steps;
  hid_t group_step;
  hid_t group_global;
  inline void ghw_H5CHK(const bool flag) { ASSERTL1((cmd) >= 0, "HDF5 ERROR"); }

  inline void write_dataspace(hid_t group, std::string key, hid_t dataspace,
                              double *value) {
    auto dataset = H5Dcreate2(group, key.c_str(), H5T_NATIVE_DOUBLE, dataspace,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ghw_H5CHK(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                       H5P_DEFAULT, value));
    ghw_H5CHK(H5Dclose(dataset));
  }
  inline void write_dataspace(hid_t group, std::string key, hid_t dataspace,
                              int *value) {
    auto dataset = H5Dcreate2(group, key.c_str(), H5T_NATIVE_INT, dataspace,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ghw_H5CHK(H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       value));
    ghw_H5CHK(H5Dclose(dataset));
  }
  inline void write_dataspace(hid_t group, std::string key, hid_t dataspace,
                              long long int *value) {
    auto dataset = H5Dcreate2(group, key.c_str(), H5T_NATIVE_LLONG, dataspace,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    ghw_H5CHK(H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       value));
    ghw_H5CHK(H5Dclose(dataset));
  }

  template <typename T>
  inline void write_group(hid_t group, std::string key, T value) {
    const hsize_t dims[1] = {1};
    auto dataspace = H5Screate_simple(1, dims, NULL);
    this->write_dataspace(group, key, dataspace, &value);
    ghw_H5CHK(H5Sclose(dataspace));
  }

public:
  GenericHDF5Writer(std::string filename)
      : filename(filename), step(-1), group_step(-1) {
    this->file = H5Fcreate(this->filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                           H5P_DEFAULT);
    ASSERTL1(this->file != H5I_INVALID_HID, "Invalid HDF5 file identifier");
    // Create the group for global data.
    std::string group_name = "global_data";
    this->group_global = H5Gcreate(this->file, group_name.c_str(), H5P_DEFAULT,
                                   H5P_DEFAULT, H5P_DEFAULT);
    group_name = "step_data";
    this->group_steps = H5Gcreate(this->file, group_name.c_str(), H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT);
  }

  template <typename T>
  inline void write_value_global(std::string key, T value) {
    this->write_group(this->group_global, key, value);
  }
  template <typename T> inline void write_value_step(std::string key, T value) {
    this->write_group(this->group_step, key, value);
  }
  inline void close() {
    ghw_H5CHK(H5Gclose(this->group_steps));
    ghw_H5CHK(H5Gclose(this->group_global));
    ghw_H5CHK(H5Fclose(this->file));
  }
  inline void step_start(int step_in = -1) {
    if (step_in > -1) {
      this->step = step_in;
    }
    // Create the group for this time step.
    std::string step_name = std::to_string(this->step++);
    this->group_step = H5Gcreate(this->group_steps, step_name.c_str(),
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  }

  inline void step_end() { ghw_H5CHK(H5Gclose(this->group_step)); }
};

#endif
