cmake_minimum_required(VERSION 3.5.1)

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
message(STATUS "Found Nektar++: version ${NEKTAR++_VERSION}")

# Set RPATH for wherever Nektar++ is installed.
set(CMAKE_INSTALL_RPATH "${NEKTAR++_LIBRARY_DIRS}")

# Add source files.
add_executable(SimpleSOL EquationSystems/SOLSystem.cpp Forcing/SourceTerms.cpp
                         SimpleSOL.cpp)

# ==================================================================================================
# Temporarily link CompressibleFlowSolver source using built object files, since
# they aren't in a Nektar library
set(NEKTAR_BUILD_SOLVERS_DIR ${NEKTAR++_ROOT_DIR}/build_tree/solvers)

file(GLOB CFSSOLVER_INSTALL_OBJS
     "${NEKTAR_BUILD_SOLVERS_DIR}/CompressibleFlowSolver/CMakeFiles/*/*/*.o")

set(CFSSOLVER_OBJS ${CFSSOLVER_BUILD_OBJS} ${CFSSOLVER_INSTALL_OBJS})

if(CFSSOLVER_OBJS STREQUAL "")
  message(
    FATAL_ERROR
      "CFSSOLVER_OBJS is empty: did cmake find the Nektar++ solvers directory?")
endif()
# ==================================================================================================

# Set target includes, libraries, compiler defs
target_compile_definitions(SimpleSOL PUBLIC ${NEKTAR++_DEFINITIONS})
target_include_directories(
  SimpleSOL PUBLIC ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS}
                   ${NEKTAR_BUILD_SOLVERS_DIR})
target_link_libraries(SimpleSOL ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES}
                      ${CFSSOLVER_OBJS})

# Install location
install(TARGETS SimpleSOL DESTINATION ${CMAKE_INSTALL_PREFIX})
