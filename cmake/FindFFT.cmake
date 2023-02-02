# FindFFT.cmake
#
# Finds an implementation of the FFTW interface. It will look for Intel MKL and,
# if that is not available,  fall back to FFTW3.
#
# This will define the following variables
#
# FFT_FOUND FFT_IMPLEMENTATION
#
# and the targets
#
# fft::fft
#

set(FFT_FOUND FALSE)

# Hack to deal with the fact Intel seems to be inconsistent with whether it
# prefers you to use dpcpp or icpx directly. MKL requires the compiler to be
# dpcpp, but the DPCPP cmake files actually prefer that icpx is used.
if(CMAKE_CXX_COMPILER MATCHES "icpx$")
  set(CMAKE_CXX_COMPILER_ORIGINAL ${CMAKE_CXX_COMPILER})
  set(CMAKE_CXX_COMPILER "dpcpp")
endif()
# Needs to be linked statically or else we get conflicts between the 32 and
# 64-bit integer index interfaces for MKL (former used by Nektar++, latter is
# needed to use the SYCL version of MKL here). See
# https://github.com/ExCALIBUR-NEPTUNE/NESO/pull/118#issuecomment-1330809280
set(MKL_LINK static)
# FIXME: Should we be setting the MPI implementation here so it can run with
# things other than intelmpi?

if(NOT NESO_DISABLE_MKL)
  if(DEFINED ENV{MKLROOT})
    # Avoids bug in MKLConfig.cmake which means it checks system directories
    # before the ones in its own installation. This can sometimes be an issue
    # when building with Spack.
    set(MKL_ROOT $ENV{MKLROOT})
  endif()
  find_package(MKL CONFIG QUIET)
endif()

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
