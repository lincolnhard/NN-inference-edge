project(primus)

cmake_minimum_required(VERSION 3.4.1)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

include_directories(src
				libraries/spdlog/include
				libraries/nlohmann-json
)

add_executable(list-device
			examples/list-device.cpp
			src/napi/builder.cpp
			src/log_stream.cpp
)

target_link_libraries(list-device
				vulkan
				android
				neuralnetworks
)

add_executable(mobilenetssd-fp32
			examples/list-device.cpp
			src/napi/builder.cpp
			src/log_stream.cpp
)

target_link_libraries(mobilenetssd-fp32
				vulkan
				android
				neuralnetworks
)

add_executable(mobilenetssd-uint8
			examples/list-device.cpp
			src/napi/builder.cpp
			src/log_stream.cpp
)

target_link_libraries(mobilenetssd-uint8
				vulkan
				android
				neuralnetworks
)

add_executable(bisenet-uint8
			examples/list-device.cpp
			src/napi/builder.cpp
			src/log_stream.cpp
)

target_link_libraries(bisenet-uint8
				vulkan
				android
				neuralnetworks
)

add_executable(espnetv2-uint8
			examples/espnetv2-uint8.cpp
			src/napi/builder.cpp
			src/log_stream.cpp
)

target_link_libraries(espnetv2-uint8
				vulkan
				android
				neuralnetworks
)
