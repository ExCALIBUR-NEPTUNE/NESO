include(CMakeDependentOption)

# find_package(Nektar++ REQUIRED) add_definitions(${NEKTAR++_DEFINITIONS}) If
# you're on a system with CMake >=3.12, also add this ADD_COMPILE_DEFINITIONS
# line: add_compile_definitions(${NEKTAR++_GENERATED_DEFINITIONS})
# include_directories(${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
# INCLUDE_DIRECTORIES(./nektar) link_directories(${NEKTAR++_LIBRARY_DIRS}
# ${NEKTAR++_TP_LIBRARY_DIRS})

# find_package(MPI REQUIRED) find_package(NESO-PARTICLES REQUIRED)

set(EXEC_TARGET_NAME MaxwellWave2D3V)

# create the solver library
set(ELECTROMAGNETIC_2D3V_LIBRARY_NAME
    MaxwellWave2D3V_ObjLib
    CACHE INTERNAL "")
add_library(${ELECTROMAGNETIC_2D3V_LIBRARY_NAME} OBJECT
            ${CMAKE_CURRENT_LIST_DIR}/EquationSystems/MaxwellWavePIC.cpp)

target_compile_options(
  ${ELECTROMAGNETIC_2D3V_LIBRARY_NAME}
  PRIVATE ${NEKTAR++_DEFINITIONS} ${NEKTAR++_GENERATED_DEFINITIONS})

target_include_directories(
  ${ELECTROMAGNETIC_2D3V_LIBRARY_NAME}
  PRIVATE ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS}
          ${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include
          ${NESO_PARTICLES_INCLUDE_PATH})

target_link_libraries(
  ${ELECTROMAGNETIC_2D3V_LIBRARY_NAME}
  PRIVATE ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES} ${MPI_LIBRARIES}
          ${NESO_PARTICLES_LIBRARIES} ${NESO_PARTICLES_LINK_FLAGS})

# create the solver binary
file(GLOB MaxwellWave2D3VSource ${CMAKE_CURRENT_LIST_DIR}/MaxwellWave2D3V.cpp)

add_executable(${EXEC_TARGET_NAME} ${MaxwellWave2D3VSource})

target_compile_options(
  ${EXEC_TARGET_NAME} PRIVATE ${NEKTAR++_DEFINITIONS}
                              ${NEKTAR++_GENERATED_DEFINITIONS})

target_link_libraries(
  ${EXEC_TARGET_NAME}
  PRIVATE ${ELECTROMAGNETIC_2D3V_LIBRARY_NAME} ${NEKTAR++_LIBRARIES}
          ${NEKTAR++_TP_LIBRARIES} ${MPI_LIBRARIES} ${NESO_PARTICLES_LIBRARIES}
          ${NESO_PARTICLES_LINK_FLAGS})

target_include_directories(
  ${EXEC_TARGET_NAME}
  PRIVATE ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS}
          ${MPI_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/include
          ${NESO_PARTICLES_INCLUDE_PATH})

add_sycl_to_target(TARGET ${EXEC_TARGET_NAME} SOURCES ${MaxwellWave2D3VSource})

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Exec rpath
set_target_properties(
  ${EXEC_TARGET_NAME} PROPERTIES INSTALL_RPATH ${INSTALL_RPATH}
                                 ${NEKTAR++_LIBRARY_DIRS})
