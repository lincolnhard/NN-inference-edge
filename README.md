# NN inference using SNPE
SNPE introduction:
[https://developer.qualcomm.com/docs/snpe/index.html](https://developer.qualcomm.com/docs/snpe/index.html)

SDK download:
[https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)

## Run on phone (835, Android)
Download [NDK]([https://developer.android.com/ndk/downloads](https://developer.android.com/ndk/downloads))
```
########## host side ##########
cd app
ndk-build
# send to target
adb push obj/local/arm64-v8a/snpe-sample /data/local/tmp
# login to target
adb shell
########## target side ##########
export LD_LIBRARY_PATH=/data/local/tmp/libsnpe:/data/local/tmp/libopencv
export ADSP_LIBRARY_PATH="/data/local/tmp/libdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
./snpe-sampe
```