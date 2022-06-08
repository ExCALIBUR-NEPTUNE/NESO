#ifndef __RUNINFO_H__
#define __RUNINFO_H__

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

class RunInfo {
public:
  RunInfo(sycl::queue &Q);

  void get_device_info(sycl::queue &Q);

/*
 * List all devices detected by SYCL
 */
  void list_all_devices();

/*
 * Get information about the selected device
 */
void get_selected_device_info(sycl::queue &Q);
};
#endif // __RUNINFO_H__
