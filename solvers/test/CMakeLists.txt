# ##############################################################################
# Find dependencies
# ##############################################################################
find_package(GTest QUIET)
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

find_package(
  Boost 1.74.0
  COMPONENTS
  REQUIRED)

# ##############################################################################
# Specify source files
# ##############################################################################
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

set(INTEGRATION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/integration)
set(INTEGRATION_SRC_FILES
    ${TEST_MAIN}
    ${CMAKE_CURRENT_SOURCE_DIR}/solver_test_utils.cpp
    ${INTEGRATION_SRC}/SimpleSOL/test_SimpleSOL.cpp
    ${INTEGRATION_SRC}/Electrostatic2D3V/TwoStream/test_two_stream.cpp
    ${INTEGRATION_SRC}/Electrostatic2D3V/ElectronBernsteinWaves/test_ebw.cpp
    ${INTEGRATION_SRC}/Electrostatic2D3V/Integrators/boris_uniform_b.cpp
    ${INTEGRATION_SRC}/DriftReduced/test_DriftReduced.cpp)

check_file_list(${INTEGRATION_SRC} cpp "${INTEGRATION_SRC_FILES}" "")

# ##############################################################################
# Set up targets
# ##############################################################################
# Remove coverage flags
list(REMOVE_ITEM BUILD_TYPE_COMPILE_FLAGS "--coverage")
list(REMOVE_ITEM TEST_LIBRARIES "gcov")

# Build the integration test suite
set(INTEGRATION_EXE solverIntegrationTests)
add_executable(${INTEGRATION_EXE} ${INTEGRATION_SRC_FILES})
target_compile_options(${INTEGRATION_EXE} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
target_include_directories(${INTEGRATION_EXE}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  ${INTEGRATION_EXE}
  PRIVATE ${NESO_LIBRARY_NAME} ${SOLVER_LIBS}
          ${ELECTROSTATIC_2D3V_LIBRARY_NAME} GTest::gmock_main GTest::gtest
          Boost::boost)
if(ENABLE_NESO_PROFILING)
  target_compile_definitions(${INTEGRATION_EXE}
                             PRIVATE NESO_PARTICLES_PROFILING_REGION)
endif()

# Define the integration test executable as a sycl target
add_sycl_to_target(TARGET ${INTEGRATION_EXE} SOURCES ${TEST_MAIN}
                   ${INTEGRATION_SRC_FILES})
# Register tests with CTest
gtest_add_tests(TARGET ${INTEGRATION_EXE})
