/*
 * Module for handling run information, such as git revision and state, uuid, and execution device.
 */

#include "run_info.hpp"

RunInfo::RunInfo(sycl::queue &Q,const char *git_revision_in, const char *git_repo_state_in) : git_revision(git_revision_in), git_repo_state(git_repo_state_in) {

	get_device_info(Q);
	report_run_info();

}

/*
 * Get information about execution devices
 */
void RunInfo::get_device_info(const sycl::queue &Q){

	get_selected_device_info(Q);

}

/*
 * List all devices detected by SYCL
 */
void RunInfo::list_all_devices(){

    std::cout << "Available devices:" << std::endl;
    for (const auto &p : sycl::platform::get_platforms()) {
      for (const auto &d : p.get_devices()) {
        std::cout << "- " << d.get_info<sycl::info::device::name>()
                  << std::endl;
      }
    }
}

/*
 * Get information about the selected device
 * Note, this information is available from the sycl
 * queue, but this form may be easier to use
 */
void RunInfo::get_selected_device_info(const sycl::queue &Q){

	device_name = Q.get_device().get_info<sycl::info::device::name>();
	device_vendor = Q.get_device().get_info<sycl::info::device::vendor>();
	device_type = "UNKNOWN";
	if( Q.get_device().is_cpu()){
		device_type = "CPU";
	} else if( Q.get_device().is_gpu()){
		device_type = "GPU";
	} else if( Q.get_device().is_accelerator()){
		device_type = "ACCELERATOR";
	}

};

/*
 * Print run information to screen
 */
void RunInfo::report_run_info(){

  // Not sure this is always useful information
  // Maybe include this in a verbose mode?
  // list_all_devices();

  // Display the device that SYCL has selected
  std::cout << "NESO running on "
              << device_name << std::endl;
  std::cout << "Vendor: "
              << device_vendor << std::endl;
  std::cout << "Type: "
              << device_type << std::endl;

  std::cout << std::endl;

  std::cout << "Git revision: " << git_revision << "\n";
  std::cout << "Git repo state: " << git_repo_state << "\n";

  std::cout << std::endl;

};
