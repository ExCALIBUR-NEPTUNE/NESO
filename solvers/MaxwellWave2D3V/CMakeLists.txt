include(CMakeDependentOption)

set(EXEC_TARGET_NAME MaxwellWave2D3V)

# create the solver library
set(MAXWELLWAVE_2D3V_LIBRARY_NAME
    MaxwellWave2D3V_ObjLib
    CACHE INTERNAL "")
add_library(${MAXWELLWAVE_2D3V_LIBRARY_NAME} OBJECT
    ${CMAKE_CURRENT_LIST_DIR}/EquationSystems/MaxwellWaveSystem.cpp)

target_include_directories(
  ${MAXWELLWAVE_2D3V_LIBRARY_NAME}
  PRIVATE ${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include
          ${NESO_PARTICLES_INCLUDE_PATH})

target_link_libraries(
  ${MAXWELLWAVE_2D3V_LIBRARY_NAME}
  PRIVATE ${NESO_LIBRARY_NAME} Nektar++::nektar++ ${MPI_LIBRARIES}
          ${NESO_PARTICLES_LIBRARIES} ${NESO_PARTICLES_LINK_FLAGS})

# create the solver binary
file(GLOB MaxwellWave2D3VSource
     ${CMAKE_CURRENT_LIST_DIR}/MaxwellWave2D3V.cpp)

add_executable(${EXEC_TARGET_NAME} ${MaxwellWave2D3VSource})

target_link_libraries(
  ${EXEC_TARGET_NAME}
  PRIVATE ${MAXWELLWAVE_2D3V_LIBRARY_NAME} Nektar++::nektar++
          ${MPI_LIBRARIES} ${NESO_PARTICLES_LIBRARIES})

add_link_options(${EXEC_TARGET_NAME} PRIVATE ${NESO_PARTICLES_LINK_FLAGS})

target_include_directories(
  ${EXEC_TARGET_NAME} PRIVATE ${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include
                              ${NESO_PARTICLES_INCLUDE_PATH})

add_sycl_to_target(TARGET ${EXEC_TARGET_NAME} SOURCES
                   ${MaxwellWave2D3VSource})

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
