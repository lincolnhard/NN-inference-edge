# NDK_TOOLCHAIN_VERSION := clang
# APP_OPTIM := release
APP_PLATFORM := android-16
APP_ABI := armeabi-v7a
APP_STL := c++_shared
APP_CPPFLAGS += -std=c++17 -fexceptions -frtti
APP_LDFLAGS = -nodefaultlibs -lc -lm -ldl -lgcc