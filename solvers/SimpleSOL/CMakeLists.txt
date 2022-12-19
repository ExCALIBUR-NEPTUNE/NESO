# Identify source files
set(SIMPLE_SOL_SRC_FILES EquationSystems/SOLSystem.cpp Forcing/SourceTerms.cpp
                         SimpleSOL.cpp)

# ================================ Exec target ================================
set(EXEC_TARGET_NAME SimpleSOL)

add_executable(${EXEC_TARGET_NAME} main.cpp ${SIMPLE_SOL_SRC_FILES})

# Compile options, target includes
target_compile_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})

# Linker options, target libs
target_link_options(${EXEC_TARGET_NAME} PRIVATE ${BUILD_TYPE_LINK_FLAGS})

target_link_libraries(${EXEC_TARGET_NAME} PRIVATE Nektar++::nektar++)
# Install location, rpath
install(TARGETS ${EXEC_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

set_target_properties(
  ${EXEC_TARGET_NAME} PROPERTIES INSTALL_RPATH ${INSTALL_RPATH}
                                 ${NEKTAR++_LIBRARY_DIRS})
