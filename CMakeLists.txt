project(aurora)

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

add_executable(${PROJECT_NAME}
			examples/main-build-nnapi-uint8.cpp
			src/napi/builder.cpp
			src/log_stream.cpp
)

target_link_libraries(${PROJECT_NAME}
				vulkan
				android
				neuralnetworks
)
