LOCAL_PATH := $(call my-dir)


include $(CLEAR_VARS)
LOCAL_MODULE := aurora-android
LOCAL_SRC_FILES := examples/main-nnapi.cpp
LOCAL_SRC_FILES += $(addprefix src/, $(notdir $(wildcard $(LOCAL_PATH)/src/*.cpp)))
LOCAL_C_INCLUDES += /Users/lincolnlee/Documents/NN-inference-edge/libraries/nlohmann-json
LOCAL_C_INCLUDES += /Users/lincolnlee/Documents/NN-inference-edge/libraries/spdlog/include
LOCAL_C_INCLUDES += /Users/lincolnlee/Documents/NN-inference-edge/src
LOCAL_SHARED_LIBRARIES := libopencv_core libopencv_highgui libopencv_imgproc libopencv_imgcodecs libopencv_videoio
LOCAL_LDLIBS := -lGLESv2 -lEGL -lneuralnetworks -landroid
include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_core
LOCAL_SRC_FILES := libraries/opencv/android/libopencv_core.so
LOCAL_EXPORT_C_INCLUDES += libraries/opencv/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_highgui
LOCAL_SRC_FILES := libraries/opencv/android/libopencv_highgui.so
LOCAL_EXPORT_C_INCLUDES += libraries/opencv/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_imgproc
LOCAL_SRC_FILES := libraries/opencv/android/libopencv_imgproc.so
LOCAL_EXPORT_C_INCLUDES += libraries/opencv/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_imgcodecs
LOCAL_SRC_FILES := libraries/opencv/android/libopencv_imgcodecs.so
LOCAL_EXPORT_C_INCLUDES += libraries/opencv/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libopencv_videoio
LOCAL_SRC_FILES := libraries/opencv/android/libopencv_videoio.so
LOCAL_EXPORT_C_INCLUDES += libraries/opencv/include
include $(PREBUILT_SHARED_LIBRARY)
