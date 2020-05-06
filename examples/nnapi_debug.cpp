#include <iostream>
#include <cassert>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include "simplemodel.h"


#define NUM_ELEM 10


bool SimpleModel::compute()
{
    // Create an ANeuralNetworksExecution object from the compiled model.
    // Note:
    //   1. All the input and output data are tied to the ANeuralNetworksExecution object.
    //   2. Multiple concurrent execution instances could be created from the same compiled model.
    // This sample only uses one execution of the compiled model.
    ANeuralNetworksExecution *execution;
    int32_t status = ANeuralNetworksExecution_create(compilation_, &execution);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Set all the elements of the first input tensor (tensor1) to the same value as inputValue1.
    // It's not a realistic example but it shows how to pass a small tensor to an execution.
    std::fill(inputTensor1_.data(), inputTensor1_.data() + NUM_ELEM, 100.0f);

    // Tell the execution to associate inputTensor1 to the model inputs.
    // Note that the index "0" here means the first operand of the modelInput list, not all operand list
    status = ANeuralNetworksExecution_setInput(execution, 0, nullptr, inputTensor1_.data(), NUM_ELEM * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Set the output tensor that will be filled by executing the model.
    // We use shared memory here to minimize the copies needed for getting the output data.
    status = ANeuralNetworksExecution_setOutputFromMemory(execution, 0, nullptr, outMem_, 0, NUM_ELEM * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Start the execution of the model.
    // Note that the execution here is asynchronous, and an ANeuralNetworksEvent object will be created to monitor the status of the execution.
    ANeuralNetworksEvent *event = nullptr;
    status = ANeuralNetworksExecution_startCompute(execution, &event);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Wait until the completion of the execution. This could be done on a different
    // thread. By waiting immediately, we effectively make this a synchronous call.
    status = ANeuralNetworksEvent_wait(event);


    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

    // Validate the results.
    float *outputTensorPtr = reinterpret_cast<float *>(mmap(nullptr, NUM_ELEM * sizeof(float), PROT_READ, MAP_SHARED, outFd_, 0));

    for (int32_t idx = 0; idx < NUM_ELEM; ++idx)
    {
        std::cout << outputTensorPtr[idx] << std::endl;
    }

    munmap(outputTensorPtr, NUM_ELEM * sizeof(float));
    return true;
}

bool SimpleModel::createCompiledModel()
{
    int32_t status;

    // Create the ANeuralNetworksModel handle.
    status = ANeuralNetworksModel_create(&model_);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    uint32_t dimensions[] = {NUM_ELEM};
    ANeuralNetworksOperandType float32TensorType{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = sizeof(dimensions) / sizeof(dimensions[0]),
            .dimensions = dimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    ANeuralNetworksOperandType scalarInt32Type{
            .type = ANEURALNETWORKS_INT32,
            .dimensionCount = 0,
            .dimensions = nullptr,
            .scale = 0.0f,
            .zeroPoint = 0,
    };


    // Add operands and operations to construct the model.
    // Operands are implicitly identified by the order in which they are added to the model,
    // starting from 0.
    // These indexes are not returned by the model_addOperand call. The application must
    // manage these values. Here, we use opIdx to do the bookkeeping.
    uint32_t opIdx = 0;

    // We first add the operand for the NONE activation function, and set its value to ANEURALNETWORKS_FUSED_NONE.
    // This constant scalar operand will be used for all 3 operations.
    status = ANeuralNetworksModel_addOperand(model_, &scalarInt32Type);
    uint32_t fusedActivationFuncNone = opIdx++;
    std::cout << "fusedActivationFuncNone: " << fusedActivationFuncNone << std::endl;
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    FuseCode fusedActivationCodeValue = ANEURALNETWORKS_FUSED_NONE;
    status = ANeuralNetworksModel_setOperandValue(model_, fusedActivationFuncNone, &fusedActivationCodeValue, sizeof(fusedActivationCodeValue));
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Add operands for the tensors.
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor0 = opIdx++;
    std::cout << "tensor0: " << tensor0 << std::endl;
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // tensor0 is a constant tensor that was established during training.
    // We read these values from the corresponding ANeuralNetworksMemory object.
    status = ANeuralNetworksModel_setOperandValueFromMemory(model_, tensor0, savedMem_, 0, NUM_ELEM * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // tensor1 is one of the user provided input tensors to the trained model.
    // Its value is determined per-execution.
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor1 = opIdx++;
    std::cout << "tensor1: " << tensor1 << std::endl;
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // tensor2 is the output of the first ADD operation.
    // Its value is computed during execution.
    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor2 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Add the first ADD operation.
    std::vector<uint32_t> add1InputOperands = {tensor0, tensor1, fusedActivationFuncNone};
    status = ANeuralNetworksModel_addOperation(model_, ANEURALNETWORKS_ADD, add1InputOperands.size(), add1InputOperands.data(), 1, &tensor2);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Identify the input and output tensors to the model.
    status = ANeuralNetworksModel_identifyInputsAndOutputs(model_, 1, &tensor1, 1, &tensor2);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Finish constructing the model.
    // The values of constant and intermediate operands cannot be altered after
    // the finish function is called.
    status = ANeuralNetworksModel_finish(model_);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Create the ANeuralNetworksCompilation object for the constructed model.
    status = ANeuralNetworksCompilation_create(model_, &compilation_);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Set the preference for the compilation, so that the runtime and drivers can make better decisions.
    // Here we prefer to get the answer quickly, so we choose ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER.
    status = ANeuralNetworksCompilation_setPreference(compilation_, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    // Finish the compilation.
    status = ANeuralNetworksCompilation_finish(compilation_);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    std::cout << "build network fin" << std::endl;
    return true;
}

SimpleModel::~SimpleModel()
{
    ANeuralNetworksCompilation_free(compilation_);
    ANeuralNetworksModel_free(model_);
    ANeuralNetworksMemory_free(savedMem_);
    ANeuralNetworksMemory_free(inMem_);
    ANeuralNetworksMemory_free(outMem_);
    close(savedinFd_);
    close(inFd_);
    close(outFd_);
}

SimpleModel::SimpleModel()
{
    model_ = nullptr;
    compilation_ = nullptr;
    inputTensor1_.resize(NUM_ELEM);

    // Create ANeuralNetworksMemory from a file containing the trained data.
    savedinFd_ = open("weight", O_RDONLY);
    int32_t status = ANeuralNetworksMemory_createFromFd(NUM_ELEM * sizeof(float), PROT_READ, savedinFd_, 0, &savedMem_);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }


    // Create ASharedMemory to hold the data for input and output tensor.
    inFd_ = ASharedMemory_create("input", NUM_ELEM * sizeof(float));
    outFd_ = ASharedMemory_create("output", NUM_ELEM * sizeof(float));

    // Create ANeuralNetworksMemory objects from the corresponding ASharedMemory objects.
    status = ANeuralNetworksMemory_createFromFd(NUM_ELEM * sizeof(float), PROT_READ, inFd_, 0, &inMem_);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }
    status = ANeuralNetworksMemory_createFromFd(NUM_ELEM * sizeof(float), PROT_READ | PROT_WRITE, outFd_, 0, &outMem_);
    if (status != ANEURALNETWORKS_NO_ERROR)
    {
        std::cout << "status: " << status << ", line: " <<__LINE__ << std::endl;
        exit(1);
    }

    std::cout << "constrctor fin" << std::endl;
}


int main()
{
    SimpleModel simo;
    simo.createCompiledModel();
    simo.compute();
    return 0;
}