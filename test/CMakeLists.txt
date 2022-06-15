cmake_minimum_required(VERSION 3.21)
project(NESO VERSION 0.0.1)

# GoogleTest requires at least C++11 DataParallel C++ requires at least C++17
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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

if(FFTW_FOUND)
  include_directories(PkgConfig::FFTW)
endif()

enable_testing()

set(EXECUTABLE testNESO)

set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

find_package(MPI REQUIRED)

# List test source files
file(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)

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
      ${TEST_NAME}
      PRIVATE ${LIBRARY_NAME}
              gtest
              lapack
              ${CUSTOM_MKL_LINK}
              ${Boost_LIBRARIES}
              ${MPI_CXX_LINK_FLAGS}
              ${MPI_CXX_LIBRARIES})
    target_include_directories(
      ${TEST_NAME}
      PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS}
              $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
              ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(
      ${TEST_NAME} PUBLIC -fsycl
                          $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  endif()

  if(FFTW_FOUND)
    target_link_libraries(
      ${TEST_NAME}
      PRIVATE gtest ${LIBRARY_NAME} PkgConfig::FFTW ${Boost_LIBRARIES}
              ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
    target_include_directories(
      ${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS}
                           PkgConfig::FFTW ${MPI_CXX_INCLUDE_PATH})
  endif()

  add_sycl_to_target(TARGET ${TEST_NAME} SOURCES ${TEST_SOURCES})
  gtest_discover_tests(${TEST_NAME})

endforeach()

set(SRC_LIST ../src/mesh.cpp ../src/species.cpp ../src/plasma.cpp
             ../src/diagnostics.cpp ../src/simulation.cpp)

# Build a global test suite
add_executable(${EXECUTABLE} ${TEST_MAIN} ${TEST_LIST} ${SRC_LIST})

if(MKL_FOUND)
  target_compile_options(
    ${EXECUTABLE} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  target_include_directories(
    ${EXECUTABLE}
    PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS}
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
            ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(
    ${EXECUTABLE}
    PRIVATE gtest
            lapack
            ${Boost_LIBRARIES}
            sycl
            ${CUSTOM_MKL_LINK}
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_CXX_LIBRARIES})
  target_compile_options(
    ${EXECUTABLE} PUBLIC -fsycl
                         $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
endif()

if(FFTW_FOUND)
  target_include_directories(
    ${EXECUTABLE} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS}
                          PkgConfig::FFTW ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(
    ${EXECUTABLE} PRIVATE gtest ${Boost_LIBRARIES} PkgConfig::FFTW
                          ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN} ${TEST_LIST}
                   ${SRC_LIST})
gtest_discover_tests(${EXECUTABLE})
