# Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

LOCAL_PATH := $(call my-dir)
SNPE_ROOT := /Users/lincolnlee/Documents/NN-inference-using-SNPE/snpe-1.35.0.698
OPENCV_ROOT := /Users/lincolnlee/Documents/NN-inference-using-SNPE/opencv

ifeq ($(TARGET_ARCH_ABI), arm64-v8a)
   ifeq ($(APP_STL), gnustl_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/aarch64-android-gcc4.9
   else ifeq ($(APP_STL), c++_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/aarch64-android-clang6.0
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else ifeq ($(TARGET_ARCH_ABI), armeabi-v7a)
   ifeq ($(APP_STL), gnustl_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/arm-android-gcc4.9
   else ifeq ($(APP_STL), c++_shared)
      SNPE_LIB_DIR := $(SNPE_ROOT)/lib/arm-android-clang6.0
   else
      $(error Unsupported APP_STL: '$(APP_STL)')
   endif
else
   $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

include $(CLEAR_VARS)
LOCAL_MODULE := snpe-sample
LOCAL_SRC_FILES := examples/main_sign_draw.cpp
LOCAL_SRC_FILES += $(addprefix src/, $(notdir $(wildcard $(LOCAL_PATH)/src/*.cpp)))
LOCAL_SHARED_LIBRARIES := libSNPE libSYMPHONYCPU libopencv_core libopencv_highgui libopencv_imgproc libopencv_imgcodecs libopencv_videoio
LOCAL_C_INCLUDES += /Users/lincolnlee/Documents/NN-inference-using-SNPE/nlohmann-json
LOCAL_C_INCLUDES += /Users/lincolnlee/Documents/NN-inference-using-SNPE/spdlog/include
LOCAL_C_INCLUDES += /Users/lincolnlee/Documents/NN-inference-using-SNPE/app/jni/src
LOCAL_LDLIBS := -lGLESv2 -lEGL
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := libSNPE
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libSNPE.so
LOCAL_EXPORT_C_INCLUDES += $(SNPE_ROOT)/include/zdl
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libSYMPHONYCPU
LOCAL_SRC_FILES := $(SNPE_LIB_DIR)/libsymphony-cpu.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_core
LOCAL_SRC_FILES := $(OPENCV_ROOT)/android/libopencv_core.so
LOCAL_EXPORT_C_INCLUDES += $(OPENCV_ROOT)/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_highgui
LOCAL_SRC_FILES := $(OPENCV_ROOT)/android/libopencv_highgui.so
LOCAL_EXPORT_C_INCLUDES += $(OPENCV_ROOT)/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_imgproc
LOCAL_SRC_FILES := $(OPENCV_ROOT)/android/libopencv_imgproc.so
LOCAL_EXPORT_C_INCLUDES += $(OPENCV_ROOT)/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_imgcodecs
LOCAL_SRC_FILES := $(OPENCV_ROOT)/android/libopencv_imgcodecs.so
LOCAL_EXPORT_C_INCLUDES += $(OPENCV_ROOT)/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_videoio
LOCAL_SRC_FILES := $(OPENCV_ROOT)/android/libopencv_videoio.so
LOCAL_EXPORT_C_INCLUDES += $(OPENCV_ROOT)/include
include $(PREBUILT_SHARED_LIBRARY)



