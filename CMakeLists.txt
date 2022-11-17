cmake_minimum_required(VERSION 3.21)

project(
  NESO
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(ENABLE_NESO_TESTS
       "Build unit tests for this project and register with ctest" ON)

# find MPI
find_package(MPI REQUIRED)

# Include hipSYCL
find_package(hipSYCL 0.9.2 QUIET)
if(NOT hipSYCL_FOUND)
  message(WARNING "hipsycl not found")
  find_package(IntelDPCPP REQUIRED)
else()
  message(STATUS "hipsycl found")
  include_directories(${hipSYCL_INCLUDE_DIRS})
  set(HIPSYCL_TARGETS "omp")
  add_definitions(-DNESO_HIPSYCL)
endif(NOT hipSYCL_FOUND)

if(IntelDPCPP_FOUND)
  message(STATUS "IntelDPCPP found")
  add_definitions(-DNESO_DPCPP)
endif(IntelDPCPP_FOUND)

# hipsycl, trisycl and computecpp all define an "add_sycl_to_target" for the
# compilation of a target
if(NOT COMMAND add_sycl_to_target)
  # Note from hipsycl: "Do not call target_sources after add_sycl_to_target or
  # dependency tracking on compiler flags will break in subtle ways"
  function(add_sycl_to_target)

  endfunction()
endif()

# Set location of custom Boost install:
find_package(
  Boost 1.74.0
  COMPONENTS
  REQUIRED)
# INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

# Find Nektar++
find_package(Nektar++ REQUIRED)
include_directories(SYSTEM ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
link_directories(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})
message(STATUS "Found Nektar++: version ${NEKTAR++_VERSION}")

# Set RPATH for wherever Nektar++ is installed.
set(CMAKE_INSTALL_RPATH "${NEKTAR++_LIBRARY_DIRS}")

# Hack to deal with the fact Intel seems to be inconsistent with whether it
# prefers you to use dpcpp or icpx directly. MKL requires the compiler to be
# dpcpp, but the DPCPP cmake files actually prefer that icpx is used.
if(CMAKE_CXX_COMPILER MATCHES "icpx$")
  set(CMAKE_CXX_COMPILER_ORIGINAL ${CMAKE_CXX_COMPILER})
  set(CMAKE_CXX_COMPILER "dpcpp")
endif()
find_package(MKL CONFIG QUIET)
if(CMAKE_CXX_COMPILER_ORIGINAL)
  set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER_ORIGINAL})
endif()
if(MKL_FOUND)

  # Documented manner to link against mkl.
  set(CUSTOM_MKL_LINK $<LINK_ONLY:MKL::MKL_DPCPP>)

  add_definitions(-DNESO_INTEL_MKL_FFT)
  message("MKL Found")
else()
  find_package(PkgConfig REQUIRED)
  pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
  add_definitions(-DNESO_FFTW_FFT)
  message("FFTW Found")
endif()

# Uncomment this code to list ALL cmake variables
# get_cmake_property(_variableNames VARIABLES) list (SORT _variableNames)
# foreach (_variableName ${_variableNames}) message(STATUS
# "${_variableName}=${${_variableName}}") endforeach()

# ##############################################################################
# Set the build type environment variable
# ##############################################################################
if(BUILD_TYPE STREQUAL "TEST")
  set(BUILD_TYPE_COMPILE_FLAGS "-g;-O0;--coverage")
  set(TEST_LIBRARIES "gcov")
elseif(BUILD_TYPE STREQUAL "DEBUG")
  set(BUILD_TYPE_COMPILE_FLAGS "-g -O0")
  set(TEST_LIBRARIES "")
elseif(BUILD_TYPE STREQUAL "RELEASE")
  set(BUILD_TYPE_COMPILE_FLAGS "-O2")
  set(TEST_LIBRARIES "")
endif()

# Get all sources that are not the main
file(GLOB LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
list(REMOVE_ITEM LIB_SRCS ${MAIN_SRC})

# Create library
set(LIBRARY_NAME nesolib)
add_library(${LIBRARY_NAME} ${LIB_SRCS})

include(src/nektar/CMakeLists.txt)
include(solvers/Electrostatic2D3V/CMakeLists.txt)

# Various sanitizers, including coverage and address sanitizer
include(cmake/Sanitizers.cmake)
enable_sanitizers(${LIBRARY_NAME})

# Get the git commit
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC NESO_REVISION)
if(NESO_REVISION STREQUAL "GITDIR-NOTFOUND")
  set(NESO_REVISION "Unknown")
endif()
message(STATUS "Git revision: ${NESO_REVISION}")
git_local_changes(NESO_GIT_STATE)
message(STATUS "Git repo state: ${NESO_GIT_STATE}")

# Build the file containing just the commit hash This will be rebuilt on every
# commit!
configure_file("${PROJECT_SOURCE_DIR}/include/revision.hpp.in"
               "${PROJECT_SOURCE_DIR}/include/revision.hpp")
# Create executable
add_executable(${PROJECT_NAME} ${MAIN_SRC})

# define the source file objects as a sycl targets
add_sycl_to_target(TARGET ${LIBRARY_NAME} SOURCES ${LIB_SRCS})

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
  set(UNUSED_ARG -Wno-error=unused-command-line-argument)
endif()
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror
                                               ${UNUSED_ARG})

# target_compile_options( ${PROJECT_NAME} PUBLIC
# $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)

# Include src folder for header files
target_include_directories(
  ${LIBRARY_NAME} PUBLIC $<INSTALL_INTERFACE:include>
                         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
if(MKL_FOUND)
  target_include_directories(
    ${LIBRARY_NAME}
    PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# Add libraries
target_link_libraries(${LIBRARY_NAME} PUBLIC ${NEKTAR++_LIBRARIES}
                                             ${NEKTAR++_TP_LIBRARIES})
if(MKL_FOUND)
  target_link_libraries(${LIBRARY_NAME} PUBLIC ${CUSTOM_MKL_LINK})
endif()
if(FFTW_FOUND)
  target_link_libraries(${LIBRARY_NAME} PUBLIC PkgConfig::FFTW)
endif()

# Add the NESO-Particles dependencies
find_package(NESO-PARTICLES REQUIRED)
target_link_libraries(${LIBRARY_NAME} PUBLIC ${NESO_PARTICLES_LIBRARIES})
target_include_directories(${LIBRARY_NAME}
                           PUBLIC ${NESO_PARTICLES_INCLUDE_PATH})

# Include test directory
if(ENABLE_NESO_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

# Link main
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARY_NAME})

# define the main executable as a sycl target
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${MAIN_SRC})

# install the header files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX})
# install the cmake files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY bin DESTINATION ${CMAKE_INSTALL_PREFIX})
