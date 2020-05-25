# CXXCL := toolchain/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-g++
CXXCL := g++
NVCXX := nvcc

CXXFLAGS += -O0 -g -std=c++14 -Wno-deprecated -Wno-deprecated-declarations

CUDA_ARCH := -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_72,code=sm_72

INCLUDES += -I src
INCLUDES += -I /usr/include/opencv4
INCLUDES += -I libraries/spdlog/include
INCLUDES += -I libraries/nlohmann-json
INCLUDES += -I /usr/local/cuda/include

DEFINES += -D NDEBUG


LDFLAGS += -L/usr/lib/aarch64-linux-gnu
LDFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_dnn
LDFLAGS += -lglog -lboost_system
LDFLAGS += -lnvinfer -lnvparsers -lcuda -lnvinfer_plugin -lnvonnxparser -lnvonnxparser_runtime
LDFLAGS += -L/usr/local/cuda-10.0/targets/aarch64-linux/lib
LDFLAGS += -lcudart
LDFLAGS += -pthread -lm

OBJROOT := obj

SRCFILES := $(wildcard src/*.cpp)
SRCFILES += $(wildcard src/nv/*.cpp)
# SRCFILES += $(wildcard src/*.cu)


# EXAMPLEFILES := examples/mobilenetv2ssd_trt_fps.cpp
EXAMPLEFILES := examples/mobilenetv2ssd_trt_debug.cpp
# EXAMPLEFILES := examples/mnasneta1fcos_trt_debug.cpp
# EXAMPLEFILES := examples/mnasneta1fcos_trt_fps.cpp
# EXAMPLEFILES := examples/uninet_trt_fps.cpp
# EXAMPLEFILES := examples/espnetv2fusion_trt_debug.cpp
# EXAMPLEFILES := examples/espnetv2fusion_trt_fps.cpp

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