include(CMakeDependentOption)

set(EXEC_TARGET_NAME Electrostatic2D3V)

# create the solver library
set(ELECTROSTATIC_2D3V_LIBRARY_NAME
    Electrostatic2D3V_ObjLib
    CACHE INTERNAL "")
add_library(${ELECTROSTATIC_2D3V_LIBRARY_NAME} OBJECT
            ${CMAKE_CURRENT_LIST_DIR}/EquationSystems/PoissonPIC.cpp)

target_include_directories(
  ${ELECTROSTATIC_2D3V_LIBRARY_NAME}
  PRIVATE ${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include
          ${NESO_PARTICLES_INCLUDE_PATH})

target_link_libraries(
  ${ELECTROSTATIC_2D3V_LIBRARY_NAME}
  PRIVATE ${NESO_LIBRARY_NAME} Nektar++::nektar++ ${MPI_LIBRARIES}
          ${NESO_PARTICLES_LIBRARIES} ${NESO_PARTICLES_LINK_FLAGS})

# create the solver binary
file(GLOB Electrostatic2D3VSource
     ${CMAKE_CURRENT_LIST_DIR}/Electrostatic2D3V.cpp)

add_executable(${EXEC_TARGET_NAME} ${Electrostatic2D3VSource})

target_link_libraries(
  ${EXEC_TARGET_NAME}
  PRIVATE ${ELECTROSTATIC_2D3V_LIBRARY_NAME} Nektar++::nektar++
          ${MPI_LIBRARIES} ${NESO_PARTICLES_LIBRARIES})

add_link_options(${EXEC_TARGET_NAME} PRIVATE ${NESO_PARTICLES_LINK_FLAGS})

target_include_directories(
  ${EXEC_TARGET_NAME} PRIVATE ${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include
                              ${NESO_PARTICLES_INCLUDE_PATH})

add_sycl_to_target(TARGET ${EXEC_TARGET_NAME} SOURCES
                   ${Electrostatic2D3VSource})

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
