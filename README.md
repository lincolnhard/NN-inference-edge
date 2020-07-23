# NN inference using SNPE

export LD_LIBRARY_PATH=/data/local/tmp/libraries

```
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=/Users/lincolnlee/Library/Android/sdk/ndk/21.3.6528147/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DANDROID_NATIVE_API_LEVEL=29 ..
make -j4
```