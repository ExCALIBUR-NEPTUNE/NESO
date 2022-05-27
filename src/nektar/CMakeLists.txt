include(CMakeDependentOption)

# set the project name
project(CombinedSolver CXX)

FIND_PACKAGE(Nektar++ REQUIRED)
ADD_DEFINITIONS(${NEKTAR++_DEFINITIONS})
#  If you're on a system with CMake >=3.12, also add this ADD_COMPILE_DEFINITIONS line:
ADD_COMPILE_DEFINITIONS(${NEKTAR++_GENERATED_DEFINITIONS})
INCLUDE_DIRECTORIES(${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
#INCLUDE_DIRECTORIES(./nektar)
LINK_DIRECTORIES(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})


find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

file(GLOB CFSSOLVER_BUILD_OBJS "${NEKTAR++_ROOT_DIR}/../solvers/CompressibleFlowSolver/CMakeFiles/*/*/*.o")
file(GLOB ADRSOLVER_BUILD_OBJS "${NEKTAR++_ROOT_DIR}/../solvers/ADRSolver/CMakeFiles/ADRSolver.dir/EquationSystems/*.o")
file(GLOB CFSSOLVER_INSTALL_OBJS "${NEKTAR++_ROOT_DIR}/solvers_objects/CompressibleFlowSolver/CMakeFiles/*/*/*.o")
file(GLOB ADRSOLVER_INSTALL_OBJS "${NEKTAR++_ROOT_DIR}/solvers_objects/ADRSolver/CMakeFiles/ADRSolver.dir/EquationSystems/*.o")

set(ADRSOLVER_OBJS ${ADRSOLVER_BUILD_OBJS} ${ADRSOLVER_INSTALL_OBJS})
set(CFSSOLVER_OBJS ${CFSSOLVER_BUILD_OBJS} ${CFSSOLVER_INSTALL_OBJS})

if (CFSSOLVER_OBJS STREQUAL "")
  message(FATAL_ERROR "CFSSOLVER_OBJS is empty: did cmake find the Nektar++ solvers directory?")
endif()
if (ADRSOLVER_OBJS STREQUAL "")
    message(FATAL_ERROR "ADRSOLVER_OBJS is empty: did cmake find the Nektar++ solvers directory?")
endif()

SET(CombinedSolverSource src/nektar/CombinedSolver/CombinedSolver.cpp)
add_executable(CombinedSolver.x ${CombinedSolverSource})
set_target_properties(CombinedSolver.x PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(CombinedSolver.x ${ADRSOLVER_OBJS} ${CFSSOLVER_OBJS} ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES} ${MPI_LIBRARIES})



