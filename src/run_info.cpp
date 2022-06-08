/*
 * Module for handling run information, such as git revision and state, uuid, and execution device.
 */

#include "run_info.hpp"

RunInfo::RunInfo(sycl::queue &Q) {

	get_device_info(Q);

}

void RunInfo::get_device_info(sycl::queue &Q){

	list_all_devices();
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
 */
void RunInfo::get_selected_device_info(sycl::queue &Q){

    // Display the device that SYCL has selected
    std::cout << "NESO running on "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

};
