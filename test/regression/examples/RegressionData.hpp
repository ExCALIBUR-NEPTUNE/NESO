#ifndef __NESO_TEST_REGRESSION_EXAMPLES_REGRESSIONDATA_H_
#define __NESO_TEST_REGRESSION_EXAMPLES_REGRESSIONDATA_H_

#include <filesystem>
#include <hdf5.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class RegressionData {
public:
  /**
   * @brief If the entity associated with ID \p loc_id is a dataset, add its
   * name to the vector pointed at by \p opdata
   *
   * \param loc_id HDF5 ID of the target entity
   * \param name Name of the target entity
   * \param info HDF5 info obj associated with the target entity
   * \param opdata Pointer to a std::vector<std::string> in which to store names
   */
  static herr_t get_dset_names(hid_t loc_id, const char *name,
                               const H5O_info_t *info, void *opdata) {
    auto dset_names = static_cast<std::vector<std::string> *>(opdata);
    if (info->type == H5O_TYPE_DATASET) {
      dset_names->push_back(name);
    }
    return 0;
  }

  RegressionData() {}

  /**
   * @brief Read a regression data file at \p fpath
   * \param fpath Path to
   * \returns 0 on succesful read, otherwise an integer error state
   */
  void read(fs::path fpath) {
    this->fpath = fpath;

    // Open file
    hid_t file_id =
        H5Fopen(this->fpath.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID) {
      this->err_state = H5I_INVALID_HID;
      this->err_msg << "H5Fopen failed. Incorrect path?";
      return;
    }

    // Open nsteps attribute
    const std::string nsteps_str("nsteps");
    hid_t attr_id = H5Aopen(file_id, nsteps_str.c_str(), H5P_DEFAULT);
    if (attr_id == H5I_INVALID_HID) {
      this->err_state = H5I_INVALID_HID;
      this->err_msg << "Error opening " << nsteps_str << " attribute";
      return;
    }

    // Read nsteps attribute
    this->err_state = H5Aread(attr_id, H5T_NATIVE_INT, &this->nsteps);
    if (this->err_state < 0) {
      this->err_msg << "Error reading " << nsteps_str << " attribute";
      return;
    }

    // Determine dataset names
    std::vector<std::string> dset_names;
    this->err_state =
        H5Ovisit(file_id, H5_INDEX_NAME, H5_ITER_NATIVE, get_dset_names,
                 static_cast<void *>(&dset_names), H5O_INFO_ALL);
    if (this->err_state < 0) {
      this->err_msg << "Error retrieving dataset names";
      return;
    }

    // Read all datasets
    for (auto &dset_name : dset_names) {
      read_dset(file_id, dset_name);
      if (this->err_state) {
        break;
      }
    }

    H5Fclose(file_id);
  }

  /**
   * @brief Read an HDF5 dataset and store result in this->dsets
   * \param file_id ID associated with an open HDF5 file
   * \param dset_name The name of the dataset
   */
  void read_dset(const hid_t &file_id, const std::string &dset_name) {
    // Open the dataset
    hid_t dset_id = H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT);

    // Init dset data storage
    this->dsets[dset_name] = std::vector<double>();

    // Check the dataset is 1D (required for now)
    hid_t dset_space = H5Dget_space(dset_id);
    int dset_num_dims = H5Sget_simple_extent_ndims(dset_space);
    if (dset_num_dims != 1) {
      this->err_msg << "Dataset " << dset_name << " is " << dset_num_dims
                    << "-dimensional; expected 1D ";
      this->err_state = dset_num_dims;
      return;
    }

    // Determine 1D length and resize target vector accordingly
    hsize_t dset_len, dset_maxdim;
    H5Sget_simple_extent_dims(dset_space, &dset_len, &dset_maxdim);
    if (dset_len <= 0) {
      this->err_msg << "Dataset " << dset_name << " appears to have size "
                    << dset_len;
      this->err_state = dset_len;
      return;
    }
    this->dsets[dset_name].resize(dset_len);

    // Read the dataset
    this->err_state = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, &this->dsets[dset_name][0]);
    if (this->err_state == H5I_INVALID_HID) {
      this->err_msg << "Error in H5Dread for dataset " << dset_name;
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const RegressionData &obj);

  /// Datasets
  std::map<std::string, std::vector<double>> dsets;

  /// (public) Error state
  herr_t err_state;

  /// Number of steps executed by solver
  int nsteps;

private:
  std::stringstream err_msg;
  fs::path fpath;
};

/**
 * @brief Override << to output error message if error state is non-zero
 * @param os an output stream
 * @param obj a RegressionData object
 * @return \p os with appropriate message appended
 */
inline std::ostream &operator<<(std::ostream &os, const RegressionData &obj) {
  if (obj.err_state) {
    os << "Valid regression data read from " << obj.fpath;
  } else {
    os << "Encountered error [" << obj.err_state
       << "] reading regression data file at" << obj.fpath << ":" << std::endl
       << obj.err_msg.str();
  }
  return os;
}
#endif