find_package(GTest REQUIRED)

if(NOT GTest_FOUND)
  include(FetchContent)
  set(INSTALL_GTEST OFF)
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

set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

set(UNIT_EXE unitTests)
set(UNIT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/unit)
set(UNIT_SRC_FILES
  ${UNIT_SRC}/test_diagnostics.cpp
  ${UNIT_SRC}/test_fft_mkl.cpp
  ${UNIT_SRC}/test_mesh.cpp
  ${UNIT_SRC}/test_plasma.cpp
  ${UNIT_SRC}/test_species.cpp
  ${UNIT_SRC}/nektar_interface/test_geometry_transport_2d.cpp
)

set(INTEGRATION_EXE integrationTests)
set(INTEGRATION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/integration)
set(INTEGRATION_SRC_FILES
  ${INTEGRATION_SRC}/test_mms.cpp
)

# options common to both fftw/mkl
# set(LINK_LIBRARIES_COMMON
#     ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES} ${Boost_LIBRARIES}
#     ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} GTest::gtest)

# set(INCLUDE_DIRECTORIES_COMMON ${CMAKE_SOURCE_DIR}/include
#                                ${Boost_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})

# Build the unittest suite
add_executable(${UNIT_EXE} ${TEST_MAIN} ${UNIT_SRC_FILES})
target_link_libraries(${UNIT_EXE} PRIVATE ${LIBRARY_NAME} GTest::gtest)
# Define the test executable as a sycl target
add_sycl_to_target(TARGET ${UNIT_EXE} SOURCES ${TEST_MAIN} ${UNIT_SRC_FILES})
# Register tests with CTest
gtest_add_tests(TARGET ${UNIT_EXE})

# Build the integration test suite
add_executable(${INTEGRATION_EXE} ${TEST_MAIN} ${INTEGRATION_SRC_FILES})
target_link_libraries(${INTEGRATION_EXE} PRIVATE ${LIBRARY_NAME} GTest::gtest Boost::boost)
# Define the test executable as a sycl target
add_sycl_to_target(TARGET ${INTEGRATION_EXE} SOURCES ${TEST_MAIN} ${INTEGRATION_SRC_FILES})

# Register tests with CTest
gtest_add_tests(TARGET ${INTEGRATION_EXE})
