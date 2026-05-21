cmake_minimum_required(VERSION 3.18)  # CMake version check
cmake_policy(SET CMP0026 NEW)
cmake_policy(SET CMP0051 NEW)

function(debug_message messages)
  message(STATUS "\n")
  message(STATUS "${messages}")
  message(STATUS "\n")
endfunction()

# Variables
set(PROJECT_NAME gpg)
set(PY_LIB_NAME _pb_gpg)
set(NUM_PRECISION 6)

option(GPG_USE_CUDA "Build CUDA accelerated GP-GOMEA backends" $ENV{GPG_USE_CUDA})
set(COMMON_SOURCE_FILES
  src/gpu/gpu_token.cpp
  src/gpu/gpu_eval_context.cpp
)

if(GPG_USE_CUDA)
  enable_language(CUDA)
  list(APPEND COMMON_SOURCE_FILES src/gpu/fitness_cuda.cu)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_FLAGS_VALGRIND
    "${CMAKE_CXX_FLAGS_VALGRIND} -g -O0 -DDEBUG -Wall -Wconversion -pedantic ${CXX_EXTRA}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG -Wall -Wconversion ${CXX_EXTRA}")
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG ${CXX_EXTRA}")

# Create project
project(${PROJECT_NAME})               
# Enable c++17 standard
set(CMAKE_CXX_STANDARD 17)  
set(PYBIND11_CPP_STANDARD -std=c++17)

### Inclusions

# Include Eigen
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
  if(NOT EIGEN3_INCLUDE_DIR)
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  endif()
else()
  if(DEFINED ENV{EIGEN3_INCLUDE_DIR})
    set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")
  endif()
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 was not found. Install Eigen3 or set EIGEN3_INCLUDE_DIR to the directory containing Eigen/.")
  endif()
endif()

# Include python libraries
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
debug_message("Found Python\n\tDIRS: ${Python_INCLUDE_DIRS}\t${Python_NumPy_INCLUDE_DIRS}\n\tLIBS: ${Python_LIBRARIES}")

# Include pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(PYBIND11_CMAKE_DIR)
    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG QUIET)
  endif()
endif()
if(NOT pybind11_FOUND)
  message(FATAL_ERROR "pybind11 was not found. Install pybind11 or set pybind11_DIR.")
endif()

### Compilation flags

# Set correct compilation flags
if(CMAKE_BUILD_TYPE STREQUAL "debug")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "release")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "valgrind")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_VALGRIND}")
endif()


# Add executable target with source files listed in SOURCE_FILES variable
add_executable(${PROJECT_NAME} src/main.cpp ${COMMON_SOURCE_FILES})

# Seek directories for compilation
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PUBLIC ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
# linking
target_link_libraries(${PROJECT_NAME} PRIVATE ${Python_LIBRARIES} Python::NumPy pybind11::headers)

# Do not check for CTRL+C from Python for the standalone executable
target_compile_definitions(${PROJECT_NAME} PUBLIC CLI=1)
target_compile_definitions(${PROJECT_NAME} PUBLIC NUM_PRECISION=${NUM_PRECISION})
if(GPG_USE_CUDA)
  target_compile_definitions(${PROJECT_NAME} PUBLIC GPG_USE_CUDA=1)
endif()

### Pybind interface (pyfe)
add_library(${PY_LIB_NAME} MODULE src/python_interface.cpp ${COMMON_SOURCE_FILES})
set_target_properties(${PY_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(${PY_LIB_NAME} PUBLIC NUM_PRECISION=${NUM_PRECISION})
if(GPG_USE_CUDA)
  target_compile_definitions(${PY_LIB_NAME} PUBLIC GPG_USE_CUDA=1)
endif()
target_include_directories(${PY_LIB_NAME} PUBLIC ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_include_directories(${PY_LIB_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PY_LIB_NAME} PRIVATE pybind11::pybind11 pybind11::headers pybind11::module pybind11::lto Python::NumPy)
pybind11_extension(${PY_LIB_NAME})
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(${PY_LIB_NAME})
endif()
