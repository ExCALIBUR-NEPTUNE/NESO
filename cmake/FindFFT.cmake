# FindFFT.cmake
#
# Finds an implementation of the FFTW interface. It will look for Intel MKL and,
# if that is not available,  fall back to FFTW3.
#
# This will define the following variables
#
#    FFT_FOUND
#    FFT_IMPLEMENTATION
#
# and the targets
#
#     fft::fft
#

set(FFT_FOUND FALSE)

# Hack to deal with the fact Intel seems to be inconsistent with whether it
# prefers you to use dpcpp or icpx directly. MKL requires the compiler to be
# dpcpp, but the DPCPP cmake files actually prefer that icpx is used.
if(CMAKE_CXX_COMPILER MATCHES "icpx$")
  set(CMAKE_CXX_COMPILER_ORIGINAL ${CMAKE_CXX_COMPILER})
  set(CMAKE_CXX_COMPILER "dpcpp")
endif()
# FIXME: Should we be setting the MPI implementation here so it can run with
# things other than intelmpi?
find_package(MKL CONFIG QUIET)
if(CMAKE_CXX_COMPILER_ORIGINAL)
  set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER_ORIGINAL})
endif()

if(MKL_FOUND)
  set(FFT_IMPLEMENTATION "Intel_MKL")
  set(FFT_FOUND TRUE)
  add_library(fft::fft ALIAS MKL::MKL_DPCPP)
else()
  find_package(PkgConfig REQUIRED)
  pkg_search_module(FFTW QUIET fftw3 IMPORTED_TARGET)
  if(FFTW_FOUND)
    set(FFT_IMPLEMENTATION fftw)
    set(FFT_FOUND TRUE)
    add_library(fft::fft ALIAS PkgConfig::FFTW)
  endif()
endif()
