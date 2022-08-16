cmake_minimum_required(VERSION 3.21)
project(NESO VERSION 0.0.1)

# GoogleTest requires at least C++11 DataParallel C++ requires at least C++17
set(CMAKE_CXX_STANDARD 17)

enable_testing()
find_package(GTest QUIET)

if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# Set location of custom Boost install:
find_package(
  Boost 1.74.0
  COMPONENTS
  REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

find_package(MKL CONFIG QUIET)
if(NOT MKL_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
endif()

if(MKL_FOUND)
  include_directories($<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

set(EXECUTABLE testNESO)

set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

find_package(MPI REQUIRED)
find_package(NESO-PARTICLES REQUIRED)

# List test source files
file(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)

# find the nektar++ cmake vars
find_package(Nektar++ REQUIRED)
add_definitions(${NEKTAR++_DEFINITIONS})
# If you're on a system with CMake >=3.12, also add this ADD_COMPILE_DEFINITIONS
# line:
add_compile_definitions(${NEKTAR++_GENERATED_DEFINITIONS})
include_directories(${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS}
                    $ENV{NEKTAR_SOURCE_DIR}/solvers)
link_directories(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})

# options common to both fftw/mkl
set(LINK_LIBRARIES_COMMON
    ${NEKTAR++_LIBRARIES}
    ${NEKTAR++_TP_LIBRARIES}
    ${Boost_LIBRARIES}
    GTest::gtest
    ${MPI_CXX_LINK_FLAGS}
    ${MPI_CXX_LIBRARIES}
    ${NESO_PARTICLES_LIBRARIES}
    ${NESO_PARTICLES_LINK_FLAGS})

set(INCLUDE_DIRECTORIES_COMMON
    ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH}
    ${NESO_PARTICLES_INCLUDE_PATH})

# Build the tests individually
include(GoogleTest)
foreach(TEST ${TEST_SRCS})
  get_filename_component(TEST_NAME ${TEST} NAME_WLE)
  message(STATUS "Found test - ${TEST_NAME}")
  set(TEST_LIST ${TEST_LIST} ${TEST_NAME})

  set(TEST_SOURCES ${TEST_MAIN} ${TEST})
  add_executable(${TEST_NAME} ${TEST_SOURCES})
  include_directories(${Boost_INCLUDE_DIRS})

  if(MKL_FOUND)
    target_link_libraries(
      ${TEST_NAME} PRIVATE ${LIBRARY_NAME} ${CUSTOM_MKL_LINK}
                           ${LINK_LIBRARIES_COMMON})
    target_include_directories(
      ${TEST_NAME}
      PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
              ${INCLUDE_DIRECTORIES_COMMON})

    target_compile_options(
      ${TEST_NAME} PUBLIC -fsycl
                          $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  endif()

  if(FFTW_FOUND)
    target_link_libraries(
      ${TEST_NAME} PRIVATE ${LIBRARY_NAME} ${FFTW_LIBRARY_DIRS}
                           ${FFTW_ABS_LIBRARIES} ${LINK_LIBRARIES_COMMON})
    target_include_directories(
      ${TEST_NAME} PRIVATE ${FFTW_INCLUDE_DIRS} ${INCLUDE_DIRECTORIES_COMMON})
  endif()

  add_sycl_to_target(TARGET ${TEST_NAME} SOURCES ${TEST_SOURCES})
  gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 600)

endforeach()

# Build a global test suite
add_executable(${EXECUTABLE} ${TEST_MAIN} ${TEST_LIST})

if(MKL_FOUND)
  target_compile_options(
    ${EXECUTABLE} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  target_include_directories(
    ${EXECUTABLE}
    PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
            ${INCLUDE_DIRECTORIES_COMMON})
  target_link_libraries(
    ${EXECUTABLE} PRIVATE ${LIBRARY_NAME} ${CUSTOM_MKL_LINK}
                          ${LINK_LIBRARIES_COMMON})

  target_compile_options(
    ${EXECUTABLE} PUBLIC -fsycl
                         $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
endif()

if(FFTW_FOUND)
  target_include_directories(
    ${EXECUTABLE} PRIVATE ${FFTW_INCLUDE_DIRS} ${INCLUDE_DIRECTORIES_COMMON})
  target_link_libraries(
    ${EXECUTABLE}
    PRIVATE ${LIBRARY_NAME} ${LINK_LIBRARIES_COMMON} ${LINK_LIBRARIES_COMMON}
            ${FFTW_LIBRARY_DIRS} ${FFTW_ABS_LIBRARIES})
endif()

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN} ${TEST_LIST}
                   ${SRC_LIST})
gtest_discover_tests(${EXECUTABLE} DISCOVERY_TIMEOUT 600)
