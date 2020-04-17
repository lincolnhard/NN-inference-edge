# CXXCL := toolchain/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-g++
CXXCL := g++

CXXFLAGS += -O0 -g -std=c++11 -funsafe-math-optimizations -ftree-vectorize -flax-vector-conversions -fstrict-aliasing

INCLUDES += -I src
# INCLUDES += -I libraries/opencv/include
INCLUDES += -I /Users/lincolnlee/Documents/opencv/installx86/include
INCLUDES += -I /Users/lincolnlee/Documents/caffe/include
INCLUDES += -I /Users/lincolnlee/Documents/OpenBLAS
INCLUDES += -I libraries/spdlog/include
INCLUDES += -I libraries/nlohmann-json

DEFINES += -D NDEBUG
DEFINES += -D CPU_ONLY

# SHAREDPATH += -Wl,-rpath,libraries/ffmpegbuild/lib

# LDFLAGS += -Llibraries/opencv/linux/lib64
LDFLAGS += -L/Users/lincolnlee/Documents/opencv/installx86/lib
LDFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_dnn
LDFLAGS += -L/Users/lincolnlee/Documents/caffe/build/lib
LDFLAGS += -lcaffe
LDFLAGS += -L/Users/lincolnlee/Documents/OpenBLAS
LDFLAGS += -lopenblas
LDFLAGS += -L/usr/local/Cellar/glog/0.3.5_3/lib
LDFLAGS += -lglog
LDFLAGS += -pthread -lm

OBJROOT := obj

SRCFILES := $(wildcard src/*.cpp)

EXAMPLEFILES := examples/mnasneta1fcos_fps.cpp

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