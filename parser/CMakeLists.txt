cmake_minimum_required(VERSION 4.2.3)
cmake_policy(SET CMP0115 NEW)

project(cuda_matrix LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Path to LLVM/Clang
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found Clang ${Clang_VERSION}")

include_directories(${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# -------------------
# CUDA executable
# -------------------
add_executable(matrix_mult matrix_mult.cu)
set_target_properties(matrix_mult PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# -------------------
# LLVM IR parser executable
# -------------------
add_executable(parser-llvm-ir parser_llvm_ir.cpp)
target_link_libraries(parser-llvm-ir PRIVATE LLVMCore LLVMSupport LLVMIRReader)

# -------------------
# Clang AST parser executable
# -------------------
add_executable(parser_ast parser_ast.cpp)
llvm_map_components_to_libnames(llvm_libs support core)
target_link_libraries(parser_ast
    PRIVATE
    ${llvm_libs}
    clangTooling
    clangFrontend
    clangASTMatchers
    clangAST
)

# -------------------
# Kernel Inspector executable (new)
# -------------------
add_executable(kernel_inspector kernel_inspector.cpp)
llvm_map_components_to_libnames(llvm_libs_inspector core irreader support)
target_link_libraries(kernel_inspector PRIVATE ${llvm_libs_inspector})

# -------------------
# Custom command to generate LLVM IR
# -------------------
add_custom_command(
    OUTPUT matrix_mult.ll
    COMMAND clang++ -x cuda ${CMAKE_CURRENT_SOURCE_DIR}/matrix_mult.cu -S -emit-llvm --cuda-gpu-arch=sm_89
    DEPENDS matrix_mult.cu
    COMMENT "Generating LLVM IR for matrix_mult.cu"
)

add_custom_target(generate_llvm ALL DEPENDS matrix_mult.ll)