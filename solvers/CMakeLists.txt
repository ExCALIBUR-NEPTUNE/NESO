# Add all subdirectories that have a CMakeLists
file(GLOB dir_list "*")
foreach(dir ${dir_list})
  if(IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif(IS_DIRECTORY ${dir} AND EXISTS ${dir}/CMakeLists.txt)
endforeach(dir ${dir_list})
