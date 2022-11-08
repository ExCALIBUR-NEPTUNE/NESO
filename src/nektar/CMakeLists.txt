# ##############################################################################
# Find dependencies
# ##############################################################################
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# ##############################################################################
# Specify needed object files
# ##############################################################################
file(
  GLOB CFSSOLVER_BUILD_OBJS
  "${NEKTAR++_ROOT_DIR}/../solvers/CompressibleFlowSolver/CMakeFiles/*/*/*.o")
file(
  GLOB
  ADRSOLVER_BUILD_OBJS
  "${NEKTAR++_ROOT_DIR}/../solvers/ADRSolver/CMakeFiles/ADRSolver.dir/EquationSystems/*.o"
)
file(
  GLOB
  CFSSOLVER_INSTALL_OBJS
  "${NEKTAR++_ROOT_DIR}/solvers_objects/CompressibleFlowSolver/CMakeFiles/*/*/*.o"
)
file(
  GLOB
  ADRSOLVER_INSTALL_OBJS
  "${NEKTAR++_ROOT_DIR}/solvers_objects/ADRSolver/CMakeFiles/ADRSolver.dir/EquationSystems/*.o"
)

set(ADRSOLVER_OBJS ${ADRSOLVER_BUILD_OBJS} ${ADRSOLVER_INSTALL_OBJS})
set(CFSSOLVER_OBJS ${CFSSOLVER_BUILD_OBJS} ${CFSSOLVER_INSTALL_OBJS})

if(CFSSOLVER_OBJS STREQUAL "")
  message(
    FATAL_ERROR
      "CFSSOLVER_OBJS is empty: did cmake find the Nektar++ solvers directory?")
endif()
if(ADRSOLVER_OBJS STREQUAL "")
  message(
    FATAL_ERROR
      "ADRSOLVER_OBJS is empty: did cmake find the Nektar++ solvers directory?")
endif()

# ##############################################################################
# Set up targets
# ##############################################################################
set(CombinedSolverSource src/nektar/CombinedSolver/CombinedSolver.cpp)
add_executable(CombinedSolver.x ${CombinedSolverSource})
target_link_libraries(
  CombinedSolver.x PRIVATE MPI::MPI_CXX
                           ${ADRSOLVER_OBJS}
                           ${CFSSOLVER_OBJS}
                           ${NEKTAR++_LIBRARIES}
                           ${NEKTAR++_TP_LIBRARIES})
target_compile_definitions(CombinedSolver.x PRIVATE ${NEKTAR++_DEFINITIONS} ${NEKTAR++_GENERATED_DEFINITIONS})
