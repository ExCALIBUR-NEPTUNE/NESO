include(CMakeDependentOption)

set(EXEC_TARGET_NAME Electrostatic2D3V)

# create the solver library
set(ELECTROSTATIC_2D3V_LIBRARY_NAME
    Electrostatic2D3V_ObjLib
    CACHE INTERNAL "")
set(ELECTROSTATIC_2D3V_LIBRARY_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/EquationSystems/PoissonPIC.cpp)
add_library(${ELECTROSTATIC_2D3V_LIBRARY_NAME} OBJECT
            ${ELECTROSTATIC_2D3V_LIBRARY_SOURCES})

target_include_directories(${ELECTROSTATIC_2D3V_LIBRARY_NAME}
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(
  ${ELECTROSTATIC_2D3V_LIBRARY_NAME}
  PRIVATE ${NESO_LIBRARY_NAME} Nektar++::nektar++
          NESO-Particles::NESO-Particles MPI::MPI_CXX)

add_sycl_to_target(TARGET ${ELECTROSTATIC_2D3V_LIBRARY_NAME} SOURCES
                   ${ELECTROSTATIC_2D3V_LIBRARY_SOURCES})

# create the solver binary
file(GLOB Electrostatic2D3VSource
     ${CMAKE_CURRENT_LIST_DIR}/Electrostatic2D3V.cpp)

add_executable(${EXEC_TARGET_NAME} ${Electrostatic2D3VSource})

target_link_libraries(
  ${EXEC_TARGET_NAME}
  PRIVATE ${ELECTROSTATIC_2D3V_LIBRARY_NAME} Nektar++::nektar++ MPI::MPI_CXX
          NESO-Particles::NESO-Particles)

target_include_directories(${EXEC_TARGET_NAME}
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_sycl_to_target(TARGET ${EXEC_TARGET_NAME} SOURCES
                   ${Electrostatic2D3VSource})

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
