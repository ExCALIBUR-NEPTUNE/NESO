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
find_package(MPI REQUIRED)

find_package(SYCL REQUIRED)
if(SYCL_FOUND)
  message(STATUS "Using SYCL implementation ${SYCL_IMPLEMENTATION}")
  if(hipSYCL_FOUND)
    include_directories(${hipSYCL_INCLUDE_DIRS})
    set(HIPSYCL_TARGETS "omp")
    add_definitions(-DNESO_HIPSYCL)
  elseif(SYLC_IMPLEMENTATION STREQUAL "IntelDPCPP")
    add_definitions(-DNESO_DPCPP)
  endif()
endif()

find_package(
  Boost 1.74.0
  COMPONENTS
  REQUIRED)

find_package(Nektar++ REQUIRED)
include_directories(SYSTEM ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
link_directories(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})
message(STATUS "Found Nektar++: version ${NEKTAR++_VERSION}")
# Set RPATH for wherever Nektar++ is installed.
set(CMAKE_INSTALL_RPATH "${NEKTAR++_LIBRARY_DIRS}")

find_package(FFT REQUIRED)
if(FFT_FOUND)
  message(STATUS "USING FFT implementation ${FFT_IMPLEMENTATION}")
  string(TOUPPER ${FFT_IMPLEMENTATION} FFT_UPPER)
  add_definitions(-DNESO_${FFT_UPPER}_FFT)
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

# ##############################################################################
# Set up targets
# ##############################################################################
# Get all sources that are not the main
file(GLOB LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
list(REMOVE_ITEM LIB_SRCS ${MAIN_SRC})

# Create library
set(LIBRARY_NAME nesolib)
add_library(${LIBRARY_NAME} ${LIB_SRCS})
enable_sanitizers(${LIBRARY_NAME})
target_include_directories(
  ${LIBRARY_NAME} PUBLIC $<INSTALL_INTERFACE:include>
                         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(${LIBRARY_NAME} PUBLIC fft::fft ${NEKTAR++_LIBRARIES}
                                             ${NEKTAR++_TP_LIBRARIES})
add_sycl_to_target(TARGET ${LIBRARY_NAME} SOURCES ${LIB_SRCS})

# Create Nektar solvers
include(src/nektar/CMakeLists.txt)

# Create executable
add_executable(${PROJECT_NAME} ${MAIN_SRC})
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
  set(UNUSED_ARG -Wno-error=unused-command-line-argument)
endif()
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror
                                               ${UNUSED_ARG})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARY_NAME})
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
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY bin DESTINATION ${CMAKE_INSTALL_PREFIX})
