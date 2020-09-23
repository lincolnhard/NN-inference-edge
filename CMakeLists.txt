cmake_minimum_required(VERSION 3.10)

project(primus LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_72,code=sm_72")

include_directories(
    src
    /usr/include/opencv4
    /usr/local/cuda/include
	libraries/spdlog/include
	libraries/nlohmann-json
    libraries/RxCpp/Rx/v2/src
)

link_directories(
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda-10.2/targets/aarch64-linux/lib
)

link_libraries(
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
    opencv_dnn
    nvinfer
    nvparsers
    cuda
    nvinfer_plugin
    nvonnxparser
    cudart
    pthread
    m
    stdc++fs
)

file(
    GLOB
    CPP_SOURCE
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/nv/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/nie/*.cpp"
)

file(
    GLOB
    CU_SOURCE
    "${CMAKE_SOURCE_DIR}/src/nie/*.cu"
)

add_library(
    cpp_objects
    OBJECT
    ${CPP_SOURCE}
)

set_target_properties(
    cpp_objects
    PROPERTIES COMPILE_FLAGS
    "-Wall -Wno-long-long -fPIC"
)

add_library(
    cu_objects
    OBJECT
    ${CU_SOURCE}
)

set_target_properties(
    cu_objects
    PROPERTIES COMPILE_FLAGS
    "--compiler-options '-Wall -Wno-long-long -fPIC'"
)

set_target_properties(
    cu_objects
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

list(
    APPEND
    EXAMPLE_FILES
    count-fps
    create-engine
    bisenet
    espnet
    mobilenetv2ssd
    multiple
)

foreach(EX ${EXAMPLE_FILES})
    add_executable(
        ${EX}
        "${CMAKE_SOURCE_DIR}/examples/${EX}.cpp"
        $<TARGET_OBJECTS:cpp_objects>
        $<TARGET_OBJECTS:cu_objects>
    )
endforeach()
