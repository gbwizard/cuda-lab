cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuda-lab LANGUAGES C CXX CUDA)

###############################################################################
# Build setup
###############################################################################
set(CMAKE_BUILD_TYPE Release)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

###############################################################################
# C/C++ flags
###############################################################################
if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb3")
    set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
else()
    message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
endif()

###############################################################################
# CUDA
###############################################################################
set(CUDA_ARCH sm_70 CACHE STRING "Set GPU architecture")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${CUDA_ARCH}")
if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xptxas=-v")
endif()

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
add_subdirectory(convolution2D)
