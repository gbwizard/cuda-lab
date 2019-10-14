set(TARGET convolution2D)

add_executable(${TARGET}
    convolution2D.cu
)

target_link_libraries(${TARGET}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)
