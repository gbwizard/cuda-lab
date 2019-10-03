cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuda-lab LANGUAGES C CXX CUDA)

###############################################################################
# C/C++ flags
###############################################################################
set(CMAKE_C_FLAGS "-O3")
set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
set(CMAKE_BUILD_TYPE Release)

###############################################################################
# CUDA
###############################################################################
set(CUDA_ARCH sm_70 CACHE STRING "Set GPU architecture")
set(CMAKE_CUDA_FLAGS "--ftz=false -arch=${CUDA_ARCH} --use_fast_math -Xptxas=-v")

###############################################################################
# Boost
###############################################################################
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_DEBUG_LIBS OFF)
set(Boost_USE_RELEASE_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.54.0 REQUIRED COMPONENTS
    program_options
)

###############################################################################
# Global includes
###############################################################################
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

# Source subdirs
add_subdirectory(matrixMul)
