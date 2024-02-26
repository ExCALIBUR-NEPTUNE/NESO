include(CMakeDependentOption)

set(EXEC_TARGET_NAME MeshPlotting)

# create the solver library
set(MESH_PLOTTING_LIBRARY_NAME
    mesh_plotting_ObjLib
    CACHE INTERNAL "")

add_library(${MESH_PLOTTING_LIBRARY_NAME} OBJECT
            ${CMAKE_CURRENT_LIST_DIR}/MeshPlottingInner.cpp)

target_include_directories(
  ${MESH_PLOTTING_LIBRARY_NAME}
  PRIVATE ${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include
          ${NESO_PARTICLES_INCLUDE_PATH})

target_link_libraries(
  ${MESH_PLOTTING_LIBRARY_NAME}
  PRIVATE ${NESO_LIBRARY_NAME} Nektar++::nektar++ ${MPI_LIBRARIES}
          ${NESO_PARTICLES_LIBRARIES} ${NESO_PARTICLES_LINK_FLAGS})

# create the solver binary
file(GLOB MeshPlottingSource ${CMAKE_CURRENT_LIST_DIR}/MeshPlotting.cpp)

add_executable(${EXEC_TARGET_NAME} ${MeshPlottingSource})

target_link_libraries(
  ${EXEC_TARGET_NAME} PRIVATE ${MESH_PLOTTING_LIBRARY_NAME} Nektar++::nektar++
                              ${MPI_LIBRARIES} ${NESO_PARTICLES_LIBRARIES})

add_link_options(${EXEC_TARGET_NAME} PRIVATE ${NESO_PARTICLES_LINK_FLAGS})

target_include_directories(
  ${EXEC_TARGET_NAME} PRIVATE ${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include
                              ${NESO_PARTICLES_INCLUDE_PATH})

add_sycl_to_target(TARGET ${EXEC_TARGET_NAME} SOURCES ${MeshPlottingSource})

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
