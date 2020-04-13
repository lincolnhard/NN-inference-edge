CXXCL := ../../toolchain/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-g++

CXXFLAGS += -Ofast -std=c++11 -march=armv8-a -mtune=cortex-a53 -fabi-version=9 -fno-math-errno -fno-signed-zeros -funsafe-math-optimizations -ftree-vectorize -flax-vector-conversions -fstrict-aliasing

INCLUDES += -I ../../libraries/autonomy_sdk_0.62.1.1.74_navinfo/ppmp/include
INCLUDES += -I ../../libraries/opencv/include

DEFINES += -D NDEBUG
DEFINES += -D ARM64V8A
DEFINES += -D ASIO_STANDALONE
DEFINES += -D_GLIBCXX_USE_CXX11_ABI=0

SHAREDPATH += -Wl,-rpath,../../libraries/autonomy_sdk_0.62.1.1.74_navinfo/ppmp-3rdparty/lib
SHAREDPATH += -Wl,-rpath,../../libraries/ffmpegbuild/lib
SHAREDPATH += -Wl,-rpath,../../toolchain/mizarroot/usr/lib64
SHAREDPATH += -Wl,-rpath,../../toolchain/mizarroot/lib64

LDFLAGS += -L../../libraries/autonomy_sdk_0.62.1.1.74_navinfo/ppmp/lib
LDFLAGS += -lPPMP
LDFLAGS += -L../../libraries/snpe-1.25.0.287/lib/aarch64-linux-gcc4.9
LDFLAGS += -lSNPE -lsymphony-cpu -lsnpe_adsp -lcalculator -lPlatformValidatorShared
LDFLAGS += -L../../libraries/opencv/linux/lib64
LDFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio
LDFLAGS += -pthread -lm
LDFLAGS += -static-libstdc++
LDFLAGS += -lstdc++

SRCDIR += ./examples/

OBJDIR += ./obj/

OBJS_LIST += main_ppmp_save.o

OBJS += $(addprefix $(OBJDIR), $(OBJS_LIST))

APP_NAME += ppmp_saveall

all: obj $(OBJS)
	$(CXXCL) $(SHAREDPATH) $(OBJS) $(LDFLAGS) -o $(APP_NAME)

$(OBJDIR)%.o: $(SRCDIR)%.cpp
	$(CXXCL) -c -pipe $(CXXFLAGS) $(DEFINES) $(INCLUDES) $< -o $@

obj:
	mkdir -p obj

.PHONY: clean obj

clean:
	rm -rf $(OBJDIR)/* $(APP_NAME)