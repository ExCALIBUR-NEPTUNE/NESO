# Identify source files
set(SIMPLE_SOL_SRC_FILES EquationSystems/SOLSystem.cpp Forcing/SourceTerms.cpp
                         SimpleSOL.cpp)

# ============================== Object library ===============================
# Put solver specific source in an object library so that tests can use it
set(LIBRARY_NAME SimpleSOL_ObjLib)
set(SOLVER_LIBS
    "${SOLVER_LIBS} ${LIBRARY_NAME}"
    CACHE INTERNAL "")
add_library(${LIBRARY_NAME} OBJECT ${SIMPLE_SOL_SRC_FILES})
target_compile_options(${LIBRARY_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
target_link_libraries(${LIBRARY_NAME} Nektar++::nektar++)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# =================================== Exec ====================================
set(EXEC_TARGET_NAME SimpleSOL)

add_executable(${EXEC_TARGET_NAME} main.cpp $<TARGET_OBJECTS:${LIBRARY_NAME}>)

# Compile options
target_compile_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})

# Linker options, target libs
target_link_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_LINK_FLAGS})
target_link_libraries(${EXEC_TARGET_NAME} PRIVATE Nektar++::nektar++)

# Install location
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Exec rpath
set_target_properties(
  ${EXEC_TARGET_NAME} PROPERTIES INSTALL_RPATH ${INSTALL_RPATH}
                                 ${NEKTAR++_LIBRARY_DIRS})
