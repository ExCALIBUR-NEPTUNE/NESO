# Reset list of solver object libraries
set(SOLVER_LIBS "" CACHE INTERNAL "")
# Add all subdirectories that have a CMakeLists
file(GLOB dir_list "*")
foreach(dir ${dir_list})
  if(IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif(IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt)
endforeach(dir ${dir_list})

# Trim leading/trailing whitespace from list of solver obj libs to stop cmake complaining
string(STRIP "${SOLVER_LIBS}" STRIPPED_SOLVER_LIBS)
set(SOLVER_LIBS "${STRIPPED_SOLVER_LIBS}" CACHE INTERNAL "")
