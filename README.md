# NN inference on RK3399pro

## Environment
[Debian 10](https://rockchips-my.sharepoint.com/personal/addy_ke_rockchips_onmicrosoft_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Faddy%5Fke%5Frockchips%5Fonmicrosoft%5Fcom%2FDocuments%2FTB%2DRK3399ProD%2Fimage%2FTB%2DRK3399ProD%5Fdebian%5FV1%2E0&originalPath=aHR0cHM6Ly9yb2NrY2hpcHMtbXkuc2hhcmVwb2ludC5jb20vOmY6L2cvcGVyc29uYWwvYWRkeV9rZV9yb2NrY2hpcHNfb25taWNyb3NvZnRfY29tL0VnVk9TaUNYaTZGS3U2TVZxMkVSbFhFQjU2NVl4ZjFMZHEtMGVZR3dNLXlRWHc_cnRpbWU9MFBWWE50VlMyRWc)

[rknn-toolkit 1.3.0](https://github.com/rockchip-linux/rknn-toolkit/tree/v1.3.0/packages)


## Execute
```
[Host side]

# It takes time to create .rknn model file. Running below scripts on RK3399pro is not recommended
unzip data.zip -d nn-inference-edge
python convert-caffe.py


[Device side]
cd nn-inference-edge
make
./primus configs/nie-mobilenetssd.json
```
