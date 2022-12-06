# FindNektar++.cmake
#
# Wrapper for finding Nektar++ installations. This imports the provided Nektar++
# CMake config file and then creates a target for it.
#
# This will define the following variables
#
#    Nektar++_FOUND
#    All other variables provided by the default Nektar++ CMake
#
# and the targets
#
#     Nektar++::nektar++
#

find_package(Nektar++ CONFIG)

if(Nektar++_FOUND AND NOT TARGET Nektar++::nektar++)
  set(NEKTAR++_TP_FILTERED_INCLUDE_DIRS ${NEKTAR++_TP_INCLUDE_DIRS})
  list(REMOVE_ITEM NEKTAR++_TP_FILTERED_INCLUDE_DIRS "include/nektar++")

  if(DEFINED $ENV{NEKTAR_SOURCE_DIR})
    set(SOLVER_INC_DIR $ENV{NEKTAR_SOURCE_DIR}/solvers)
  endif()

  add_library(Nektar++::nektar++ INTERFACE IMPORTED)
  target_link_libraries(Nektar++::nektar++ INTERFACE ${NEKTAR++_LIBRARIES})
  target_compile_definitions(
    Nektar++::nektar++ INTERFACE ${NEKTAR++_DEFINITIONS}
                                 ${NEKTAR++_GENERATED_DEFINITIONS})
  target_include_directories(
    Nektar++::nektar++
    INTERFACE ${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_FILTERED_INCLUDE_DIRS}
              ${SOLVER_INC_DIR})
endif()
