set(TARGET matrixMul)

add_executable(${TARGET}
    matrixMul.cu
)

target_link_libraries(${TARGET}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)
