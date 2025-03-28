option(NESO_BUILD_SOLVER_TESTS "Builds tests for the demo solvers" ON)

# Reset list of solver object libraries
set(SOLVER_LIBS
    ""
    CACHE INTERNAL "")

# Add all solver subdirectories (must have a CMakeLists)
file(GLOB solver_dirs "*")
# Exclude tests
list(FILTER solver_dirs EXCLUDE REGEX ".*/test$")
foreach(dir ${solver_dirs})
  if(EXISTS ${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif(EXISTS ${dir}/CMakeLists.txt)
endforeach(dir ${solver_dirs})

# Trim leading/trailing whitespace from list of solver obj libs to stop cmake
# complaining
string(STRIP "${SOLVER_LIBS}" STRIPPED_SOLVER_LIBS)
set(SOLVER_LIBS
    "${STRIPPED_SOLVER_LIBS}"
    CACHE INTERNAL "")

# Build tests
if(NESO_BUILD_SOLVER_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()
