CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)

PROJECT(SOLSolver)

# Nektar++ requires C++11.
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

# Default install location: build/dist
IF (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "" FORCE)
ENDIF()

# Find Nektar++
FIND_PACKAGE(Nektar++ REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
LINK_DIRECTORIES(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})
MESSAGE(STATUS "Found Nektar++: version ${NEKTAR++_VERSION}")

# Set RPATH for wherever Nektar++ is installed.
SET(CMAKE_INSTALL_RPATH "${NEKTAR++_LIBRARY_DIRS}")

# Add source files.
ADD_EXECUTABLE(SOLSolver
    src/CompressibleFlowSolver/Misc/EquationOfState.cpp
    src/CompressibleFlowSolver/Misc/IdealGasEoS.cpp
    src/CompressibleFlowSolver/Misc/VariableConverter.cpp
    src/CompressibleFlowSolver/RiemannSolvers/CompressibleSolver.cpp
    src/CompressibleFlowSolver/RiemannSolvers/ExactSolverToro.cpp
    src/EquationSystems/SOLSystem.cpp
    src/Forcing/SourceTerms.cpp
    src/SOLSolver.cpp
    )

# Add target link libraries.
TARGET_LINK_LIBRARIES(SOLSolver ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES})
INSTALL(TARGETS SOLSolver DESTINATION ${CMAKE_INSTALL_PREFIX})

TARGET_COMPILE_DEFINITIONS(SOLSolver PUBLIC ${NEKTAR++_DEFINITIONS})
