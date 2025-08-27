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

set(UNIT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/unit)
set(UNIT_SRC_FILES
    ${TEST_MAIN}
    ${UNIT_SRC}/particle_utility/test_position_distribution.cpp
    ${UNIT_SRC}/particle_utility/test_particle_initialisation_line.cpp
    ${UNIT_SRC}/nektar_interface/test_composite_interaction.cpp
    ${UNIT_SRC}/nektar_interface/test_composite_function.cpp
    ${UNIT_SRC}/nektar_interface/test_exit_tolerances.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_function_evaluation.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_function_evaluation_sub_group.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_function_projection_sub_group.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_function_evaluation_3d.cpp
    ${UNIT_SRC}/nektar_interface/test_parameter_store.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_function_projection.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_function_projection_3d.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_geometry_interface.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_geometry_interface_3d.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_geometry_interface_3d_curved.cpp
    ${UNIT_SRC}/nektar_interface/test_basis_evaluation.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_embed_mapping.cpp
    ${UNIT_SRC}/nektar_interface/test_kernel_basis_evaluation.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_mapping.cpp
    ${UNIT_SRC}/nektar_interface/test_utility_cartesian_mesh.cpp
    ${UNIT_SRC}/nektar_interface/test_particle_reader.cpp
    ${UNIT_SRC}/test_solver_callback.cpp)

check_file_list(${UNIT_SRC} cpp "${UNIT_SRC_FILES}" "")

set(INTEGRATION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/integration)
set(INTEGRATION_SRC_FILES
    ${TEST_MAIN}
    ${INTEGRATION_SRC}/nektar_interface/test_composite_interaction.cpp
    ${INTEGRATION_SRC}/nektar_interface/test_particle_advection.cpp
    ${INTEGRATION_SRC}/nektar_interface/test_function_projection_order.cpp
    ${INTEGRATION_SRC}/nektar_interface/test_function_projection_order_3d.cpp)

check_file_list(${INTEGRATION_SRC} cpp "${INTEGRATION_SRC_FILES}" "")

# ##############################################################################
# Set up targets
# ##############################################################################
# Remove coverage flags
list(REMOVE_ITEM BUILD_TYPE_COMPILE_FLAGS "--coverage")
list(REMOVE_ITEM TEST_LIBRARIES "gcov")

# Build the unit test suite
set(UNIT_EXE unitTests)
add_executable(${UNIT_EXE} ${UNIT_SRC_FILES})
target_compile_options(${UNIT_EXE} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
target_link_libraries(${UNIT_EXE} PRIVATE ${NESO_LIBRARY_NAME} GTest::gtest)
if(ENABLE_NESO_PROFILING)
  target_compile_definitions(${UNIT_EXE}
                             PRIVATE NESO_PARTICLES_PROFILING_REGION)
endif()
add_sycl_to_target(TARGET ${UNIT_EXE} SOURCES ${UNIT_SRC_FILES})
gtest_add_tests(TARGET ${UNIT_EXE})

# Build the integration test suite
set(INTEGRATION_EXE integrationTests)
add_executable(${INTEGRATION_EXE} ${INTEGRATION_SRC_FILES})
target_compile_options(${INTEGRATION_EXE} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
target_include_directories(${INTEGRATION_EXE} PRIVATE)

target_link_libraries(
  ${INTEGRATION_EXE} PRIVATE ${NESO_LIBRARY_NAME} GTest::gmock_main
                             GTest::gtest Boost::boost)
if(ENABLE_NESO_PROFILING)
  target_compile_definitions(${INTEGRATION_EXE}
                             PRIVATE NESO_PARTICLES_PROFILING_REGION)
endif()

# Define the integration test executable as a sycl target
add_sycl_to_target(TARGET ${INTEGRATION_EXE} SOURCES ${TEST_MAIN}
                   ${INTEGRATION_SRC_FILES})
# Register tests with CTest
gtest_add_tests(TARGET ${INTEGRATION_EXE})
