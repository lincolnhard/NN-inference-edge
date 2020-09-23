CXXCL := g++

CXXFLAGS := -O3 -std=c++17

INCLUDES := -I src
INCLUDES += -I /usr/local/include
INCLUDES += -I /usr/include/rockchip
INCLUDES += -I libraries/spdlog/include
INCLUDES += -I libraries/nlohmann-json

# DEFINES := -D NDEBUG

LDFLAGS := -L/lib64
LDFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio
LDFLAGS += -lrknn_api
LDFLAGS += -pthread -lm

OBJROOT := obj

SRCFILES := $(wildcard src/*.cpp)
SRCFILES += $(wildcard src/rknn/*.cpp)



EXAMPLEFILES := examples/main-run.cpp


OBJS := $(addprefix $(OBJROOT)/, $(patsubst %.cpp, %.o, $(SRCFILES) $(EXAMPLEFILES)))

# $(error LHH: '$(OBJS)')


APP_NAME := primus




all: obj $(OBJS)
	$(CXXCL) $(SHAREDPATH) $(OBJS) $(LDFLAGS) -o $(APP_NAME)

$(OBJROOT)/%.o: %.cpp
	$(CXXCL) -c -pipe $(CXXFLAGS) $(DEFINES) $(INCLUDES) $< -o $@

obj:
	mkdir -vp $(OBJROOT) $(dir $(OBJS))

.PHONY: clean obj

clean:
	rm -rf $(OBJROOT) $(APP_NAME)