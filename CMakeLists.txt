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

link_libraries(vulkan
				android
				neuralnetworks
)

set(SRC_COMMON
	src/napi/builder.cpp
	src/log_stream.cpp
)

add_executable(list-device
			examples/list-device.cpp
			${SRC_COMMON}
)

add_executable(bisenet-uint8
			examples/bisenet-uint8.cpp
			${SRC_COMMON}
)

add_executable(mobilenetssd-uint8
			examples/mobilenetssd-uint8.cpp
			${SRC_COMMON}
)

add_executable(espnetv2-uint8
			examples/espnetv2-uint8.cpp
			${SRC_COMMON}
)
