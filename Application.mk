NDK_TOOLCHAIN_VERSION := clang
APP_OPTIM := release
APP_PLATFORM := android-28
APP_ABI := arm64-v8a
APP_STL := c++_shared
APP_CPPFLAGS += -std=c++17 -fexceptions -frtti
APP_LDFLAGS = -nodefaultlibs -lc -lm -ldl -lgcc