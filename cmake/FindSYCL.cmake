# FindSYCL.cmake
#
# Finds a SYCL implementation. This is currently only a thin layer over the
# CMake files provided by each implementation. It will search first for hipSYCL
# and then for Intel's DPCPP.
#
# This will define the following variables
#
#    SYCL_FOUND
#    SYCL_IMPLEMENTATION
#
# and the functions
#
#     add_sycl_to_target
#

set(SYCL_FOUND FALSE)

list(APPEND candidates "hipSYCL" "IntelDPCPP")
list(APPEND versions "0.9.2" "")

foreach(candidate version IN ZIP_LISTS candidates versions)
  if(version)
    find_package(${candidate} ${version} QUIET)
  else()
    find_package(${candidate} QUIET)
  endif()

  if(${candidate}_FOUND)
    set(SYCL_FOUND TRUE)
    set(SYCL_IMPLEMENTATION ${candidate})
    break()
  endif()
endforeach()

# hipsycl, trisycl and computecpp all define an "add_sycl_to_target" for the
# compilation of a target
if(SYCL_FOUND AND NOT COMMAND add_sycl_to_target)
  function(add_sycl_to_target)
    # TODO: set compiler/linker flags for DPCPP; do these need to be public?
  endfunction()
endif()
