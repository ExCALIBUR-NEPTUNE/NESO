cmake_minimum_required(VERSION 3.21)

project(
  NESO
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(ENABLE_NESO_TESTS
       "Build unit tests for this project and register with ctest" ON)

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
    ${SRC_DIR}/diagnostics.cpp ${SRC_DIR}/mesh.cpp ${SRC_DIR}/plasma.cpp
    ${SRC_DIR}/simulation.cpp ${SRC_DIR}/species.cpp)
set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
set(HEADER_FILES
    ${INC_DIR}/custom_types.hpp
    ${INC_DIR}/diagnostics.hpp
    ${INC_DIR}/fft_fftw.hpp
    ${INC_DIR}/fft_mkl.hpp
    ${INC_DIR}/fft_wrappers.hpp
    ${INC_DIR}/mesh.hpp
    ${INC_DIR}/plasma.hpp
    ${INC_DIR}/simulation.hpp
    ${INC_DIR}/species.hpp
    ${INC_DIR}/velocity.hpp
    ${INC_DIR}/nektar_interface/geometry_transport_2d.hpp)

# Create library
set(LIBRARY_NAME nesolib)
add_library(${LIBRARY_NAME} ${LIB_SRC_FILES} ${HEADER_FILES})
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

# Include test directory
if(ENABLE_NESO_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

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
