# Identify source files
set(BENCH_SRC_FILES main_evaluation.cpp)

# ============================== Object library ===============================
# Put solver specific source in an object library so that tests can use it
set(BENCH_LIBRARY_NAME Bench_objlib)
add_library(${BENCH_LIBRARY_NAME} OBJECT ${BENCH_SRC_FILES})
target_compile_options(${BENCH_LIBRARY_NAME}
                       PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
target_link_libraries(${BENCH_LIBRARY_NAME} PRIVATE Nektar++::nektar++
                                                    ${NESO_LIBRARY_NAME})
# Top-level include dir for nektar-interface headers
set(TOPLEVEL_INCLUDE_DIR "${INC_DIR}")
target_include_directories(
  ${BENCH_LIBRARY_NAME}
  PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${NESO_PARTICLES_INCLUDE_PATH}
         ${TOPLEVEL_INCLUDE_DIR})

add_sycl_to_target(TARGET ${BENCH_LIBRARY_NAME} SOURCES ${BENCH_SRC_FILES})

# =================================== Exec ====================================
set(EXEC_TARGET_NAME NESOBench)

add_executable(${EXEC_TARGET_NAME} main.cpp
                                   $<TARGET_OBJECTS:${BENCH_LIBRARY_NAME}>)

# Compile options
target_compile_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})

# Linker options, target libs
target_link_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_LINK_FLAGS})
target_link_libraries(
  ${EXEC_TARGET_NAME} PRIVATE Nektar++::nektar++ ${NESO_PARTICLES_LIBRARIES}
                              ${NESO_LIBRARY_NAME})

add_sycl_to_target(TARGET ${EXEC_TARGET_NAME} SOURCES main.cpp)

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Exec rpath
set_target_properties(
  ${EXEC_TARGET_NAME} PROPERTIES INSTALL_RPATH ${INSTALL_RPATH}
                                 ${NEKTAR++_LIBRARY_DIRS})
