cmake_minimum_required(VERSION 3.5.1)

project(SimpleSOL)

# Nektar++ requires C++14.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default install location: build/dist
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      ${CMAKE_BINARY_DIR}/dist
      CACHE PATH "" FORCE)
endif()

# Find Nektar++
find_package(Nektar++ REQUIRED)
include_directories(SYSTEM ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
link_directories(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})
message(STATUS "Found Nektar++: version ${NEKTAR++_VERSION}")

# Set RPATH for wherever Nektar++ is installed.
set(CMAKE_INSTALL_RPATH "${NEKTAR++_LIBRARY_DIRS}")

# Add source files.
add_executable(
  SimpleSOL
  CompressibleFlowSolver/Misc/EquationOfState.cpp
  CompressibleFlowSolver/Misc/IdealGasEoS.cpp
  CompressibleFlowSolver/Misc/VariableConverter.cpp
  CompressibleFlowSolver/RiemannSolvers/CompressibleSolver.cpp
  CompressibleFlowSolver/RiemannSolvers/ExactSolverToro.cpp
  EquationSystems/SOLSystem.cpp
  Forcing/SourceTerms.cpp
  SimpleSOL.cpp)

# Add target link libraries.
target_link_libraries(SimpleSOL ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES})
install(TARGETS SimpleSOL DESTINATION ${CMAKE_INSTALL_PREFIX})

target_compile_definitions(SimpleSOL PUBLIC ${NEKTAR++_DEFINITIONS})
