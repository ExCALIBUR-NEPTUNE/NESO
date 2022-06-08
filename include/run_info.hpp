#ifndef __RUNINFO_H__
#define __RUNINFO_H__

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

class RunInfo {
public:
  RunInfo(sycl::queue &Q,const char *git_revision, const char *git_repo_state);

  std::string device_name;
  std::string device_vendor;
  std::string device_type;
  const char *git_revision;
  const char *git_repo_state;

/*
 * Get information about execution devices
 */
  void get_device_info(const sycl::queue &Q);

/*
 * List all devices detected by SYCL
 */
  void list_all_devices();

/*
 * Get information about the selected device
 */
void get_selected_device_info(const sycl::queue &Q);

/*
 * Print run information to screen
 */
void report_run_info();

};
#endif // __RUNINFO_H__
