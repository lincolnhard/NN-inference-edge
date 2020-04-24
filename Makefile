# CXXCL := toolchain/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-g++
CXXCL := g++

CXXFLAGS += -O3 -std=c++11 -funsafe-math-optimizations -ftree-vectorize -flax-vector-conversions -fstrict-aliasing

INCLUDES += -I src
INCLUDES += -I /usr/include/opencv4
INCLUDES += -I libraries/spdlog/include
INCLUDES += -I libraries/nlohmann-json
INCLUDES += -I /usr/local/cuda/include

DEFINES += -D NDEBUG


LDFLAGS += -L/usr/lib/aarch64-linux-gnu
LDFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_dnn
LDFLAGS += -lglog -lboost_system
LDFLAGS += -lnvinfer -lnvparsers -lcuda -lnvinfer_plugin
LDFLAGS += -L/usr/local/cuda-10.0/targets/aarch64-linux/lib
LDFLAGS += -lcudart
LDFLAGS += -pthread -lm

OBJROOT := obj

SRCFILES := $(wildcard src/*.cpp)

# EXAMPLEFILES := examples/mobilenetv2unet_tensorrt.cpp
# EXAMPLEFILES := examples/combine.cpp
# EXAMPLEFILES := examples/mnasneta1fcos_trt_debug.cpp
# EXAMPLEFILES := examples/mnasneta1fcos_trt_fps.cpp
EXAMPLEFILES := examples/uninet_trt_fps.cpp

OBJS := $(addprefix $(OBJROOT)/, $(patsubst %.cpp, %.o, $(SRCFILES) $(EXAMPLEFILES)))

# $(error LHH: '$(OBJS)')

APP_NAME := aurora





all: obj $(OBJS)
	$(CXXCL) $(SHAREDPATH) $(OBJS) $(LDFLAGS) -o $(APP_NAME)

$(OBJROOT)/%.o: %.cpp
	$(CXXCL) -c -pipe $(CXXFLAGS) $(DEFINES) $(INCLUDES) $< -o $@

obj:
	mkdir -vp $(OBJROOT) $(OBJROOT)/src $(OBJROOT)/examples

.PHONY: clean obj

clean:
	rm -rf $(OBJROOT) $(APP_NAME)