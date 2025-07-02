cmake_minimum_required(VERSION 3.24)

project(
  NESO
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(ENABLE_NESO_PROFILING "Enable built-in profiling" OFF)
option(NESO_BUILD_SOLVERS "Build the built-in solvers" ON)
option(NESO_BUILD_TESTS
       "Build unit tests for this project and register with ctest" ON)
option(NESO_BUILD_TOOLS "Builds supplemental tools." ON)

# Various sanitizers, including coverage and address sanitizer
include(cmake/Sanitizers.cmake)
include(cmake/CheckFileList.cmake)

# ##############################################################################
# Find dependencies
# ##############################################################################

# USE -DNESO_SYCL_VENDOR to chose implementation at gen time
set(NESO_SYCL_VENDOR
    "Default"
    CACHE STRING "SYCL vendor to look for - required
if present")
set(NESO_SYCL_VERSION
    ""
    CACHE STRING "SYCL version to look for")
include(cmake/SYCL.cmake)
if(NESO_SYCL_VENDOR MATCHES "Default")
  find_sycl()
else()
  find_sycl(${NESO_SYCL_VENDOR} ${NESO_SYCL_VERSION})
endif()
#
# Use a custom FindNektar++ script to vide an interface target
find_package(Nektar++ REQUIRED)

# Add the NESO-Particles dependencies
find_package(NESO-Particles REQUIRED)
# Alternativly just use as submodule option(ENABLE_NESO_PARTICLES_TESTS OFF)
# add_subdirectory(neso-particles)
# include(neso-particles/cmake/restrict-keyword.cmake)

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
message(STATUS BUILD_TYPE_COMPILE_FLAGS: ${BUILD_TYPE_COMPILE_FLAGS})
message(STATUS BUILD_TYPE_LINK_FLAGS: ${BUILD_TYPE_LINK_FLAGS})

# ##############################################################################
# Set up targets
# ##############################################################################
# Get all sources that are not the main
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_SRC_FILES
    ${SRC_DIR}/nektar_interface/basis_reference.cpp
    ${SRC_DIR}/nektar_interface/composite_interaction/composite_collections.cpp
    ${SRC_DIR}/nektar_interface/composite_interaction/composite_intersection.cpp
    ${SRC_DIR}/nektar_interface/composite_interaction/composite_utility.cpp
    ${SRC_DIR}/nektar_interface/composite_interaction/composite_transport.cpp
    ${SRC_DIR}/nektar_interface/expansion_looping/jacobi_coeff_mod_basis.cpp
    ${SRC_DIR}/nektar_interface/function_evaluation.cpp
    ${SRC_DIR}/nektar_interface/function_projection.cpp
    ${SRC_DIR}/nektar_interface/geometry_transport/geometry_transport_2d.cpp
    ${SRC_DIR}/nektar_interface/geometry_transport/geometry_transport_3d.cpp
    ${SRC_DIR}/nektar_interface/geometry_transport/halo_extension.cpp
    ${SRC_DIR}/nektar_interface/particle_boundary_conditions.cpp
    ${SRC_DIR}/nektar_interface/particle_cell_mapping/map_particles_2d.cpp
    ${SRC_DIR}/nektar_interface/particle_cell_mapping/map_particles_2d_regular.cpp
    ${SRC_DIR}/nektar_interface/particle_cell_mapping/map_particles_3d.cpp
    ${SRC_DIR}/nektar_interface/particle_cell_mapping/map_particles_3d_regular.cpp
    ${SRC_DIR}/nektar_interface/particle_cell_mapping/map_particles_common.cpp
    ${SRC_DIR}/nektar_interface/particle_cell_mapping/map_particles_host.cpp
    ${SRC_DIR}/nektar_interface/particle_cell_mapping/nektar_graph_local_mapper.cpp
    ${SRC_DIR}/nektar_interface/particle_cell_mapping/x_map_bounding_box.cpp
    ${SRC_DIR}/nektar_interface/utilities.cpp
    ${SRC_DIR}/nektar_interface/solver_base/partsys_base.cpp
    ${SRC_DIR}/nektar_interface/solver_base/particle_reader.cpp)

# set(LIB_SRC_FILES_IGNORE ${SRC_DIR}/main.cpp)
check_file_list(${SRC_DIR} cpp "${LIB_SRC_FILES}" "${LIB_SRC_FILES_IGNORE}")

set(HEADER_FILES
    ${INC_DIR}/io/generic_hdf5_writer.hpp
    ${INC_DIR}/nektar_interface/basis_evaluation.hpp
    ${INC_DIR}/nektar_interface/basis_reference.hpp
    ${INC_DIR}/nektar_interface/bary_interpolation/bary_evaluation.hpp
    ${INC_DIR}/nektar_interface/bounding_box_intersection.hpp
    ${INC_DIR}/nektar_interface/cell_id_translation.hpp
    ${INC_DIR}/nektar_interface/coordinate_mapping.hpp
    ${INC_DIR}/nektar_interface/composite_interaction/composite_collection.hpp
    ${INC_DIR}/nektar_interface/composite_interaction/composite_collections.hpp
    ${INC_DIR}/nektar_interface/composite_interaction/composite_interaction.hpp
    ${INC_DIR}/nektar_interface/composite_interaction/composite_intersection.hpp
    ${INC_DIR}/nektar_interface/composite_interaction/composite_utility.hpp
    ${INC_DIR}/nektar_interface/composite_interaction/composite_transport.hpp
    ${INC_DIR}/nektar_interface/composite_interaction/line_plane_intersection.hpp
    ${INC_DIR}/nektar_interface/composite_interaction/line_line_intersection.hpp
    ${INC_DIR}/nektar_interface/solver_base/eqnsys_base.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/basis_evaluate_base.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/expansion_looping.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/geom_to_expansion_builder.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/hexahedron.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/jacobi_coeff_mod_basis.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/jacobi_expansion_looping_interface.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/prism.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/pyramid.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/quadrilateral.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/tetrahedron.hpp
    ${INC_DIR}/nektar_interface/expansion_looping/triangle.hpp
    ${INC_DIR}/nektar_interface/function_bary_evaluation.hpp
    ${INC_DIR}/nektar_interface/function_basis_evaluation.hpp
    ${INC_DIR}/nektar_interface/function_basis_projection.hpp
    ${INC_DIR}/nektar_interface/function_evaluation.hpp
    ${INC_DIR}/nektar_interface/function_projection.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/geometry_packing_utility.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/geometry_container_3d.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/geometry_local_remote_3d.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/geometry_transport.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/geometry_transport_2d.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/geometry_transport_3d.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/geometry_types_3d.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/utility_geometry.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/halo_extension.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/packed_geom_2d.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/packed_geoms_2d.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/remote_geom.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/remote_geom_2d.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/remote_geom_3d.hpp
    ${INC_DIR}/nektar_interface/geometry_transport/shape_mapping.hpp
    ${INC_DIR}/nektar_interface/parameter_store.hpp
    ${INC_DIR}/nektar_interface/particle_boundary_conditions.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/coarse_lookup_map.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/coarse_mappers_base.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/generated_linear/hexahedron.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/generated_linear/linear_newton_implementation.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/generated_linear/prism.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/generated_linear/pyramid.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/generated_linear/quadrilateral.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/generated_linear/quadrilateralembed3d.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/generated_linear/tetrahedron.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/map_particles_2d.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/map_particles_2d_regular.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/map_particles_3d.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/map_particles_3d_regular.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/map_particles_common.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/map_particles_host.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/map_particles_newton.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/mapping_newton_iteration_base.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/nektar_graph_local_mapper.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_geom_interfaces.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_generic_3d.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_hex.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_prism.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_pyr.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_quad.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_quad_embed_3d.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_relative_exit_tolerances.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_triangle_embed_3d.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/newton_tet.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/particle_cell_mapping.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/particle_cell_mapping_2d.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/particle_cell_mapping_3d.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/particle_cell_mapping_common.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/particle_cell_mapping_newton.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/x_map_bounding_box.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/x_map_newton.hpp
    ${INC_DIR}/nektar_interface/particle_cell_mapping/x_map_newton_kernel.hpp
    ${INC_DIR}/nektar_interface/particle_interface.hpp
    ${INC_DIR}/nektar_interface/particle_mesh_interface.hpp
    ${INC_DIR}/nektar_interface/special_functions.hpp
    ${INC_DIR}/nektar_interface/solver_base/empty_partsys.hpp
    ${INC_DIR}/nektar_interface/solver_base/particle_reader.hpp
    ${INC_DIR}/nektar_interface/solver_base/partsys_base.hpp
    ${INC_DIR}/nektar_interface/solver_base/time_evolved_eqnsys_base.hpp
    ${INC_DIR}/nektar_interface/typedefs.hpp
    ${INC_DIR}/nektar_interface/utilities.hpp
    ${INC_DIR}/nektar_interface/utility_mesh.hpp
    ${INC_DIR}/nektar_interface/utility_mesh_cartesian.hpp
    ${INC_DIR}/nektar_interface/utility_mesh_plotting.hpp
    ${INC_DIR}/nektar_interface/utility_sycl.hpp
    ${INC_DIR}/particle_utility/particle_initialisation_line.hpp
    ${INC_DIR}/particle_utility/position_distribution.hpp
    ${INC_DIR}/solvers/helpers/implicit_helper.hpp
    ${INC_DIR}/solvers/solver_callback_handler.hpp
    ${INC_DIR}/solvers/solver_runner.hpp)

set(HEADER_FILES_IGNORE ${INC_DIR}/revision.hpp)
check_file_list(${INC_DIR} hpp "${HEADER_FILES}" "${HEADER_FILES_IGNORE}")

# Create library
set(NESO_LIBRARY_NAME nesolib)
add_library(${NESO_LIBRARY_NAME} ${LIB_SRC_FILES} ${HEADER_FILES})
enable_sanitizers(${NESO_LIBRARY_NAME})
target_include_directories(
  ${NESO_LIBRARY_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(${NESO_LIBRARY_NAME} PUBLIC -D${SYCL_FLAG})
target_compile_options(${NESO_LIBRARY_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
target_link_options(${NESO_LIBRARY_NAME} PUBLIC ${BUILD_TYPE_LINK_FLAGS})
target_link_libraries(
  ${NESO_LIBRARY_NAME}
  PUBLIC Nektar++::nektar++ NESO-Particles::NESO-Particles
  PRIVATE ${TEST_LIBRARIES})
if(ENABLE_NESO_PROFILING)
  target_compile_definitions(${NESO_LIBRARY_NAME}
                             PUBLIC NESO_PARTICLES_PROFILING_REGION)
endif()
add_sycl_to_target(TARGET ${NESO_LIBRARY_NAME} SOURCES ${LIB_SRC_FILES})

# find MPI
find_package(MPI REQUIRED)

# Solvers
if(NESO_BUILD_SOLVERS)
  add_subdirectory(solvers)
endif()

# Tools
if(NESO_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

# Include test directory
if(NESO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

# ##############################################################################
# Configure installation
# ##############################################################################
set(TARGETS_FILE NESOTargets)
set(INSTALL_CONFIGDIR lib/cmake/${PROJECT_NAME})

install(
  TARGETS ${NESO_LIBRARY_NAME}
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
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindNektar++.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SYCL.cmake
  DESTINATION ${INSTALL_CONFIGDIR})

# ##############################################################################
# Export the build tree
# ##############################################################################
export(
  EXPORT ${TARGETS_FILE}
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_cmake/${TARGETS_FILE}.cmake"
  NAMESPACE NESO::)
export(PACKAGE ${PROJECT_NAME})
