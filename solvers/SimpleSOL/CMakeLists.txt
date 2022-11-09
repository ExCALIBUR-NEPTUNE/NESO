CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)

PROJECT(SimpleSOL)

# Nektar++ requires C++14.
SET(CMAKE_CXX_STANDARD 14)
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
ADD_EXECUTABLE(SimpleSOL
    CompressibleFlowSolver/Misc/EquationOfState.cpp
    CompressibleFlowSolver/Misc/IdealGasEoS.cpp
    CompressibleFlowSolver/Misc/VariableConverter.cpp
    CompressibleFlowSolver/RiemannSolvers/CompressibleSolver.cpp
    CompressibleFlowSolver/RiemannSolvers/ExactSolverToro.cpp
    EquationSystems/SOLSystem.cpp
    Forcing/SourceTerms.cpp
    SimpleSOL.cpp
    )

# Add target link libraries.
TARGET_LINK_LIBRARIES(SimpleSOL ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES})
INSTALL(TARGETS SimpleSOL DESTINATION ${CMAKE_INSTALL_PREFIX})

TARGET_COMPILE_DEFINITIONS(SimpleSOL PUBLIC ${NEKTAR++_DEFINITIONS})
