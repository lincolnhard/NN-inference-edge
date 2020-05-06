LOCAL_PATH := $(call my-dir)


include $(CLEAR_VARS)
LOCAL_MODULE := aurora-android
LOCAL_SRC_FILES := examples/nnapi_debug.cpp
# LOCAL_SRC_FILES += $(addprefix src/, $(notdir $(wildcard $(LOCAL_PATH)/src/*.cpp)))
# LOCAL_SHARED_LIBRARIES := libSNPE libSYMPHONYCPU libopencv_core libopencv_highgui libopencv_imgproc libopencv_imgcodecs libopencv_videoio
# LOCAL_C_INCLUDES += /Users/lincolnlee/Documents/NN-inference-using-SNPE/nlohmann-json
# LOCAL_C_INCLUDES += /Users/lincolnlee/Documents/NN-inference-using-SNPE/spdlog/include

LOCAL_C_INCLUDES += $(LOCAL_PATH)/src/nnapi
LOCAL_LDLIBS := -lGLESv2 -lEGL -lneuralnetworks -landroid
include $(BUILD_EXECUTABLE)

