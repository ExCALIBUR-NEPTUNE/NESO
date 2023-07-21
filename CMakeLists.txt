cmake_minimum_required(VERSION 3.21)

project(
  NESO
  VERSION 0.0.1
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(ENABLE_NESO_TESTS
       "Build unit tests for this project and register with ctest" OFF)

# Various sanitizers, including coverage and address sanitizer
include(cmake/Sanitizers.cmake)

# ##############################################################################
# Find dependencies
# ##############################################################################
find_package(SYCL REQUIRED)
if(SYCL_FOUND)
  message(STATUS "Using SYCL implementation ${SYCL_IMPLEMENTATION}")
  if(hipSYCL_FOUND)
    set(HIPSYCL_TARGETS "omp")
    set(SYCL_FLAG NESO_HIPSYCL)
  elseif(SYCL_IMPLEMENTATION STREQUAL "IntelDPCPP")
    set(SYCL_FLAG NESO_DPCPP)
  endif()
endif()

# Use a custom FindNektar++ script to provide an interface target
find_package(Nektar++ REQUIRED)

find_package(FFT REQUIRED)
if(FFT_FOUND)
  message(STATUS "USING FFT implementation ${FFT_IMPLEMENTATION}")
  string(TOUPPER ${FFT_IMPLEMENTATION} FFT_UPPER)
  set(FFT_FLAG NESO_${FFT_UPPER}_FFT)
endif()

# Uncomment this code to list ALL cmake variables
# get_cmake_property(_variableNames VARIABLES) list (SORT _variableNames)
# foreach (_variableName ${_variableNames}) message(STATUS
# "${_variableName}=${${_variableName}}") endforeach()

# ##############################################################################
# Configure git commit information for build
# ##############################################################################
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
               "${PROJECT_BINARY_DIR}/revision.hpp")

# ##############################################################################
# Set the build type environment variable
# ##############################################################################
set(CMAKE_BUILD_TYPE
    RELEASE
    CACHE STRING "Specifies the compile flags to be used.")

if(CMAKE_BUILD_TYPE STREQUAL "TEST")
  set(BUILD_TYPE_COMPILE_FLAGS "-g;-O0;--coverage")
  set(BUILD_TYPE_LINK_FLAGS "--coverage")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(TEST_LIBRARIES "gcov")
  else()
    set(TEST_LIBRARIES "")
  endif()
elseif(BUILD_TYPE STREQUAL "DEBUG")
  set(BUILD_TYPE_COMPILE_FLAGS "-g -O0")
  set(BUILD_TYPE_LINK_FLAGS "")
  set(TEST_LIBRARIES "")
elseif(BUILD_TYPE STREQUAL "RELEASE")
  set(BUILD_TYPE_COMPILE_FLAGS "-O2")
  set(BUILD_TYPE_LINK_FLAGS "")
  set(TEST_LIBRARIES "")
endif()

# ##############################################################################
# Set up targets
# ##############################################################################
# Get all sources that are not the main
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_SRC_FILES
    ${SRC_DIR}/diagnostics.cpp
    ${SRC_DIR}/mesh.cpp
    ${SRC_DIR}/plasma.cpp
    ${SRC_DIR}/run_info.cpp
    ${SRC_DIR}/simulation.cpp
    ${SRC_DIR}/species.cpp
    ${SRC_DIR}/linear_interpolator_1D.cpp)
set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
set(HEADER_FILES
    ${INC_DIR}/custom_types.hpp
    ${INC_DIR}/diagnostics.hpp
    ${INC_DIR}/fft_fftw.hpp
    ${INC_DIR}/fft_mkl.hpp
    ${INC_DIR}/fft_wrappers.hpp
    ${INC_DIR}/mesh.hpp
    ${INC_DIR}/plasma.hpp
    ${INC_DIR}/run_info.hpp
    ${INC_DIR}/simulation.hpp
    ${INC_DIR}/species.hpp
    ${INC_DIR}/velocity.hpp
    ${INC_DIR}/nektar_interface/geometry_transport_2d.hpp
    ${INC_DIR}/atomic_data_reader.hpp
    ${INC_DIR}/csv_atomic_data_reader.hpp
    ${INC_DIR}/interpolator.hpp
    ${INC_DIR}/linear_interpolator_1D.hpp)

# Create library
set(LIBRARY_NAME nesolib)
add_library(${LIBRARY_NAME} SHARED ${LIB_SRC_FILES} ${HEADER_FILES})
enable_sanitizers(${LIBRARY_NAME})
target_include_directories(
  ${LIBRARY_NAME} PUBLIC $<INSTALL_INTERFACE:include>
                         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(${LIBRARY_NAME} PUBLIC -D${SYCL_FLAG} -D${FFT_FLAG})
target_compile_options(${LIBRARY_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
target_link_options(${LIBRARY_NAME} PUBLIC ${BUILD_TYPE_LINK_FLAGS})
target_link_libraries(
  ${LIBRARY_NAME}
  PUBLIC Nektar++::nektar++ fft::fft
  PRIVATE ${TEST_LIBRARIES})
add_sycl_to_target(TARGET ${LIBRARY_NAME} SOURCES ${LIB_SRC_FILES})

# Create executable
add_executable(${PROJECT_NAME} ${MAIN_SRC})
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
  set(UNUSED_ARG -Wno-error=unused-command-line-argument)
endif()
target_compile_options(
  ${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror ${UNUSED_ARG}
                          ${BUILD_TYPE_COMPILE_FLAGS})
target_link_options(${PROJECT_NAME} PRIVATE ${BUILD_TYPE_LINK_FLAGS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARY_NAME} ${TEST_LIBRARIES})
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${MAIN_SRC})

# find MPI
find_package(MPI REQUIRED)
# Add the NESO-Particles dependencies
find_package(NESO-PARTICLES REQUIRED)

target_link_libraries(${LIBRARY_NAME} PUBLIC ${NESO_PARTICLES_LIBRARIES})
target_include_directories(${LIBRARY_NAME}
                           PUBLIC ${NESO_PARTICLES_INCLUDE_PATH})

# Solvers
add_subdirectory(solvers)

# Include test directory
if(ENABLE_NESO_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

# ##############################################################################
# Add cppyy python bindings
# ##############################################################################

# Get include files
file(GLOB LIB_HPPS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)# ${CMAKE_CURRENT_SOURCE_DIR}/include/nektar_interface/*.hpp)

set(CMAKE_MODULE_PATH ${CPPYY_MODULE_PATH} ${CMAKE_MODULE_PATH})
set(Cppyy_DIR ${CPPYY_MODULE_PATH})
find_package(Cppyy)

set(HIPSYCL_INC
    /home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/hipsycl-0.9.2-nfvpn6i7t3b25zbzgw3lpx35yuo3khbq/include/
)

message(${LIB_HPPS})

# Specification at https://cppyy.readthedocs.io/en/latest/cmake_interface.html
cppyy_add_bindings(
  "PyNESO"
  "0.0.1"
  "Joseph Parker"
  "joseph.parker@ichec.ie"
  LANGUAGE_STANDARD
  "17"
  H_FILES
  ${LIB_HPPS}
  GENERATE_OPTIONS
  -D${SYCL_FLAG}
  -I
  ${HIPSYCL_INC}
  -DBINDING_BUILD=on
  -pthread
  -std=c++1z
  -m64
  -I/home/jparker/code/NESO/venv_3_9_12/lib/python3.9/site-packages/cppyy_backend/include
  INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  ${MPI_CXX_INCLUDE_PATH}
  $<INSTALL_INTERFACE:include>
  ${HIPSYCL_INC}
  /home/jparker/code/NESO/include
  /home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/fftw-3.3.10-u6j4gsbk2z3s6iosn32aw3crgsitva5v/include
  LINK_LIBRARIES
  ${BUILD_TYPE_LINK_FLAGS})

# ##############################################################################
# Configure installation
# ##############################################################################
set(TARGETS_FILE NESOTargets)
set(INSTALL_CONFIGDIR lib/cmake/${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME} ${LIBRARY_NAME}
  EXPORT ${TARGETS_FILE}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/include/
  TYPE INCLUDE
  PATTERN "*.hpp.in" EXCLUDE)

# Install information on the exported targets
install(
  EXPORT ${TARGETS_FILE}
  NAMESPACE NESO::
  DESTINATION ${INSTALL_CONFIGDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_cmake/NESOConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/NESOConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_cmake/NESOConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

# Install the cmake files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_cmake/NESOConfigVersion.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_cmake/NESOConfig.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindFFT.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindSYCL.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindNektar++.cmake
  DESTINATION ${INSTALL_CONFIGDIR})

# ##############################################################################
# Export the build tree
# ##############################################################################
export(
  EXPORT ${TARGETS_FILE}
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_cmake/${TARGETS_FILE}.cmake"
  NAMESPACE NESO::)
export(PACKAGE ${PROJECT_NAME})
