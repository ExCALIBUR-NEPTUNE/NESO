include(CMakeDependentOption)

find_package(Nektar++ REQUIRED)
add_definitions(${NEKTAR++_DEFINITIONS})
# If you're on a system with CMake >=3.12, also add this ADD_COMPILE_DEFINITIONS
# line:
add_compile_definitions(${NEKTAR++_GENERATED_DEFINITIONS})
include_directories(${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
# INCLUDE_DIRECTORIES(./nektar)
link_directories(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})

find_package(MPI REQUIRED)
find_package(NESO-PARTICLES REQUIRED)

include_directories(${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include
                    ${NESO_PARTICLES_INCLUDE_PATH})

set(ELECTROSTATIC_2D3V_LIBRARY_NAME Electrostatic2D3V)
add_library(${ELECTROSTATIC_2D3V_LIBRARY_NAME}
            ${CMAKE_CURRENT_LIST_DIR}/EquationSystems/PoissonPIC.cpp)

file(GLOB Electrostatic2D3VSource
     ${CMAKE_CURRENT_LIST_DIR}/Electrostatic2D3V.cpp)

add_executable(Electrostatic2D3V.x ${Electrostatic2D3VSource})
set_target_properties(Electrostatic2D3V.x PROPERTIES COMPILE_FLAGS
                                                     "${MPI_COMPILE_FLAGS}")
target_link_libraries(
  Electrostatic2D3V.x
  PRIVATE ${ELECTROSTATIC_2D3V_LIBRARY_NAME} ${NEKTAR++_LIBRARIES}
          ${NEKTAR++_TP_LIBRARIES} ${MPI_LIBRARIES} ${NESO_PARTICLES_LIBRARIES}
          ${NESO_PARTICLES_LINK_FLAGS})

add_sycl_to_target(TARGET Electrostatic2D3V.x SOURCES
                   ${Electrostatic2D3VSource})
