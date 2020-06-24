# CXXCL := toolchain/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-g++
CXXCL := g++
NVCXX := nvcc

CXXFLAGS := -O3 -std=c++14

CUDA_ARCH := -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_72,code=sm_72

INCLUDES := -I src
INCLUDES += -I /usr/include/opencv4
INCLUDES += -I libraries/spdlog/include
INCLUDES += -I libraries/nlohmann-json
INCLUDES += -I /usr/local/cuda/include

DEFINES := -D NDEBUG

LDFLAGS += -pthread -lm -lprotobuf

OBJROOT := obj

SRCFILES := $(wildcard src/*.cpp)
SRCFILES += $(wildcard src/onnx/*.cpp)



EXAMPLEFILES := examples/main-parse-onnx.cpp


OBJS := $(addprefix $(OBJROOT)/, $(patsubst %.cu, %.o, $(patsubst %.cpp, %.o, $(SRCFILES) $(EXAMPLEFILES))))

# $(error LHH: '$(OBJS)')


APP_NAME := aurora




all: obj $(OBJS)
	$(CXXCL) $(SHAREDPATH) $(OBJS) $(LDFLAGS) -o $(APP_NAME)

$(OBJROOT)/%.o: %.cpp
	$(CXXCL) -c -pipe $(CXXFLAGS) $(DEFINES) $(INCLUDES) $< -o $@
$(OBJROOT)/%.o: %.cu
	$(NVCXX) -c $(CXXFLAGS) $(CUDA_ARCH) $(DEFINES) $(INCLUDES) $< -o $@

obj:
	mkdir -vp $(OBJROOT) $(dir $(OBJS))

.PHONY: clean obj

clean:
	rm -rf $(OBJROOT) $(APP_NAME)