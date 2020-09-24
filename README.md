# NN Edge Inference

master branch for TX2

nnapi branch for G90

rknn branch for RK3399

TensorRT engines please download [here](https://gallopwave-my.sharepoint.com/:u:/p/lincoln_lee/EVpG3vto2RlPmSo7leUNr5QB-oSksz-4DTxEiqjzK8d1kw?e=1TezlE) and extract under this repo


```
mkdir build
cd build
cmake ..
make -j8

# Create tensorRT engine (optional)
./create-engine ../configs/mobilenetv2ssd.json

# Calculate fps
./count-fps ../configs/mobilenetv2ssd.json

# Inference
./espnet ../configs/espnet.json
./mobilenetv2ssd ../configs/mobilenetv2ssd.json
./bisenet ../configs/bisenet.json

# test nie mobilenet + bisenet fps
./multiple ../configs/bisenet_mobilenet.json
```


