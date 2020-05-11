#include <iostream>
#include <cassert>
#include <android/NeuralNetworks.h>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include <string>
#include <unistd.h>
#include <fcntl.h>

#define NET_W 640
#define NET_H 480
#define NET_C 3

#define CHECK_NNAPI_ERROR(status)                                                                   \
        if (status != ANEURALNETWORKS_NO_ERROR)                                                     \
        {                                                                                           \
            std::cout << "Error status: " << status << ", line: " <<__LINE__ << std::endl;          \
            exit(1);                                                                                \
        }


class EspNetV2Model {
public:
    explicit EspNetV2Model();
    ~EspNetV2Model();

    bool createModel();
    bool compute();

private:
    ANeuralNetworksModel *model_;
    ANeuralNetworksCompilation *compilation_;
    
    ANeuralNetworksMemory *savedMem_;

    ANeuralNetworksMemory *inMem_;
    ANeuralNetworksMemory *outMem_;

    // std::vector<float> inputTensor1_;
    int savedinFd_;
    int inFd_;
    int outFd_;
};

EspNetV2Model::~EspNetV2Model()
{
    ANeuralNetworksModel_free(model_);
    // close(inFd_);
    close(outFd_);
    // ANeuralNetworksMemory_free(inMem_);
    ANeuralNetworksMemory_free(outMem_);
}

EspNetV2Model::EspNetV2Model()
{
    int32_t status = 0;
    status = ANeuralNetworksModel_create(&model_);
    CHECK_NNAPI_ERROR(status);

    // Create and assign memory to hold the data for input and output tensor.
    // inFd_ = ASharedMemory_create("input", NET_W * NET_H * NET_C * sizeof(float));
    outFd_ = ASharedMemory_create("output", NET_W * NET_H * 32 * sizeof(float));
    // status = ANeuralNetworksMemory_createFromFd(NET_W * NET_H * NET_C * sizeof(float), PROT_READ, inFd_, 0, &inMem_);
    // CHECK_NNAPI_ERROR(status);
    status = ANeuralNetworksMemory_createFromFd(NET_W * NET_H * 32 * sizeof(float), PROT_READ | PROT_WRITE, outFd_, 0, &outMem_);
    CHECK_NNAPI_ERROR(status);



    // Create ANeuralNetworksMemory from a file containing the trained data.
    savedinFd_ = open("espnetv2weight", O_RDONLY);
    status = ANeuralNetworksMemory_createFromFd(3 * 32 * 3 * 3 * sizeof(float), PROT_READ, savedinFd_, 0, &savedMem_);
    CHECK_NNAPI_ERROR(status);






    ANeuralNetworksOperandType inTensorOp{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 4,
            .dimensions = {1, 480, 640, 3},
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    ANeuralNetworksOperandType outTensorOp{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 4,
            .dimensions = {1, 480, 640, 32},
            .scale = 0.0f,
            .zeroPoint = 0,
    };


    ANeuralNetworksOperandType convWeightOp{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = 4,
            .dimensions = {32, 3, 3, 3},
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    // start building
    uint32_t opIdx = 0;



    status = ANeuralNetworksModel_addOperand(model_, &inTensorOp);
    CHECK_NNAPI_ERROR(status);
    uint32_t inputtensor = opIdx++;

    status = ANeuralNetworksModel_addOperand(model_, &outTensorOp);
    CHECK_NNAPI_ERROR(status);
    uint32_t outputtensor = opIdx++;





    status = ANeuralNetworksModel_addOperand(model_, &convWeightOp);
    CHECK_NNAPI_ERROR(status);
    uint32_t Conv_0 = opIdx++;



    // Conv_0 is a constant tensor that was established during training.
    // We read these values from the corresponding ANeuralNetworksMemory object.
    status = ANeuralNetworksModel_setOperandValueFromMemory(model_, Conv_0, savedMem_, 0, 3 * 3 * 3 * 32 * sizeof(float));
    CHECK_NNAPI_ERROR(status);


    std::vector<uint32_t> convoperands = {inputtensor, Conv_0};
    status = ANeuralNetworksModel_addOperation(model_, ANEURALNETWORKS_CONV_2D, convoperands.size(), convoperands.data(), 1, &outputtensor);
    CHECK_NNAPI_ERROR(status);




    std::cout << "constrctor fin" << std::endl;
}







int main()
{
    EspNetV2Model simo;
    // simo.createCompiledModel();
    // simo.compute();
    return 0;
}