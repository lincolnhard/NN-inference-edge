#include <iostream>
#include "napi/builder.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "log_stream.hpp"

#include <chrono>
#include <thread>


static auto LOG = spdlog::stdout_color_mt("MAIN");

int main(int ac, char *av[])
{
    int32_t deviceIndex = -1;
    if (ac == 1)
    {
        deviceIndex = -1;
    }
    else if (ac == 2)
    {
        deviceIndex = std::stoi(std::string(av[1]));
    }
    else
    {
        SLOG_ERROR << "Usage: " << av[0] << " [device ID] (0: GPU, 1: APU, 2: CPU, -1: Auto-select)" << std::endl;
        return 1;
    }

    gallopwave::ModelBuilder builder;
	SLOG_INFO << "2" << std::endl;
    builder.getSdkVersion();
    builder.getDevices();

    // NNAPI default data layout NHWC
    const uint32_t NET_WIDTH = 640;
    const uint32_t NET_HEIGHT = 480;
    const uint32_t NET_CHANNELS = 3;
    const uint32_t NET_IN_SIZE = NET_WIDTH * NET_HEIGHT * NET_CHANNELS;

    // input
    uint8_t *indataptr = new uint8_t[NET_IN_SIZE];
    std::fill(indataptr, indataptr + NET_IN_SIZE, 1);

    // kernels
    const uint32_t DUMMY_N = 1280;
    const uint32_t DUMMY_C = 100;
    const uint32_t DUMMY_H = 10;
    const uint32_t DUMMY_W = 10;
    const uint32_t DUMMY_KERNEL_SIZE = DUMMY_N * DUMMY_H * DUMMY_W * DUMMY_C;
    uint8_t *weightptr = new uint8_t[DUMMY_KERNEL_SIZE];
    std::fill(weightptr, weightptr + DUMMY_KERNEL_SIZE, 2);
    int32_t *biasptr = new int32_t[DUMMY_N];
    std::fill(biasptr, biasptr + DUMMY_N, 1);

    // start to build
	//layer: base_net.level1.conv.weight
    builder.addTensor("data", {1, NET_HEIGHT, NET_WIDTH, NET_CHANNELS}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.addTensor("conv1_weight", {32, 3, 3, 3}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
    builder.addTensor("conv1_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv1", "data", "conv1_weight", "conv1_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, false, ANEURALNETWORKS_FUSED_RELU6, "conv1_out");

    builder.avgpool("avg_pool1", "conv1_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 2, 2, 4, 4, 
	                ANEURALNETWORKS_FUSED_NONE, "avg_pool1_out");
                    
    //layer: base_net.level2_0.eesp.proj_1x1.conv.weight
	builder.addTensor("conv2_weight", {8, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv2_bias", {8}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv2", "conv1_out", "conv2_weight", "conv2_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv2_out");
                    
	//layer: base_net.level2_0.eesp.spp_dw.0.conv.weight
	builder.addTensor("conv2_1_weight", {1, 3, 3, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv2_1_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv2_1", "conv2_out", "conv2_1_weight", "conv2_1_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv2_1_out");
					
	//layer: base_net.level2_0.eesp.spp_dw.1.conv.weight
	builder.addTensor("conv2_2_weight", {1, 3, 3, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv2_2_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv2_2", "conv2_out", "conv2_2_weight", "conv2_2_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv2_2_out");
					
	//layer: base_net.level2_0.eesp.spp_dw.0.conv.weight    
	builder.addTensor("conv2_3_weight", {1, 3, 3, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv2_3_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv2_3", "conv2_out", "conv2_3_weight", "conv2_3_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv2_3_out");
					
	//layer: base_net.level2_0.eesp.spp_dw.0.conv.weight
	builder.addTensor("conv2_4_weight", {1, 3, 3, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv2_4_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv2_4", "conv2_out", "conv2_4_weight", "conv2_4_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv2_4_out");
    //name: 585, 587, 589                 
    builder.eltwise("add_1_layer2", "conv2_1_out", "conv2_2_out", ANEURALNETWORKS_FUSED_NONE, "add_1_layer2_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_2_layer2", "add_1_layer2_out", "conv2_3_out", ANEURALNETWORKS_FUSED_NONE, "add_2_layer2_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_3_layer2", "add_2_layer2_out", "conv2_4_out", ANEURALNETWORKS_FUSED_NONE, "add_3_layer2_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    
    builder.addTensor("pointwise2_1_weight", {32, 1, 1, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise2_1_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise2_1", "conv2_1_out", "pointwise2_1_weight", "pointwise2_1_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise2_1_out"
                   );
                    
    builder.addTensor("pointwise2_2_weight", {32, 1, 1, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise2_2_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise2_2", "conv2_2_out", "pointwise2_2_weight", "pointwise2_2_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise2_2_out"
                   );
                    
    builder.addTensor("pointwise2_3_weight", {32, 1, 1, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise2_3_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise2_3", "conv2_3_out", "pointwise2_3_weight", "pointwise2_3_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise2_3_out"
                   );
                    
    builder.addTensor("pointwise2_4_weight", {32, 1, 1, 8}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise2_4_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise2_4", "conv2_4_out", "pointwise2_4_weight", "pointwise2_4_bias",           
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise2_4_out"
                   );
    //name: 590               
    builder.eltwise("add_pointwise2_1", "pointwise2_1_out", "pointwise2_2_out", ANEURALNETWORKS_FUSED_NONE, 
                    "add_pointwise2_1_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_pointwise2_2", "add_pointwise2_1_out", "pointwise2_3_out", ANEURALNETWORKS_FUSED_NONE, 
                    "add_pointwise2_2_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_pointwise2_3", "add_pointwise2_2_out", "pointwise2_4_out", ANEURALNETWORKS_FUSED_RELU6, 
                    "add_pointwise2_3_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    
    //name: 594                
    builder.addTensor("pointwise2_5_weight", {32, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise2_5_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise2_5", "add_pointwise2_3_out", "pointwise2_5_weight", "pointwise2_5_bias",           
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise2_5_out"
                   );
    //name: 596           
    builder.eltwise("add_pointwise2_4", "avg_pool1_out", "pointwise2_5_out", ANEURALNETWORKS_FUSED_NONE, 
                    "add_pointwise2_4_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    //name: 596
    builder.addTensor("pointwise2_6_weight", {64, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise2_6_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise2_6", "add_pointwise2_4_out", "pointwise2_6_weight", "pointwise2_6_bias",           
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise2_6_out"
                   );
    //name: 598
    builder.avgpool("avg_pool2", "data", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 2, 2, 4, 4, 
	                ANEURALNETWORKS_FUSED_NONE, "avg_pool2_out");
    //name: 600
    builder.avgpool("avg_pool3", "avg_pool2_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 2, 2, 4, 4, 
	                ANEURALNETWORKS_FUSED_NONE, "avg_pool3_out");   
    //name: 601
    builder.addTensor("conv2_5_weight", {3, 3, 3, 3}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv2_5_bias", {3}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv2_5", "avg_pool3_out", "conv2_5_weight", "conv2_5_bias",           
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv2_5_out");
    //name: 605
    builder.addTensor("conv2_6_weight", {64, 1, 1, 3}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv2_6_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv2_6", "conv2_5_out", "conv2_6_weight", "conv2_6_bias",           
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv2_6_out");
    //name: 607           
    builder.eltwise("add_4_layer2", "pointwise2_6_out", "conv2_6_out", ANEURALNETWORKS_FUSED_RELU6, 
                    "add_4_layer2_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    //name: 611
    builder.avgpool("avg_pool4", "add_4_layer2_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 2, 2, 4, 4, 
	                ANEURALNETWORKS_FUSED_NONE, "avg_pool4_out");
    ///////////////////////
    //layer: base_net.level2_0.eesp.proj_1x1.conv.weight
	builder.addTensor("conv3_weight", {16, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv3_bias", {16}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv3", "add_4_layer2_out", "conv3_weight", "conv3_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv3_out");
                    
	//layer: base_net.level2_0.eesp.spp_dw.0.conv.weight
	builder.addTensor("conv3_1_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv3_1_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv3_1", "conv3_out", "conv3_1_weight", "conv3_1_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv3_1_out");
					
	//layer: base_net.level2_0.eesp.spp_dw.1.conv.weight
	builder.addTensor("conv3_2_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv3_2_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv3_2", "conv3_out", "conv3_2_weight", "conv3_2_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv3_2_out");
					
	//layer: base_net.level2_0.eesp.spp_dw.0.conv.weight    
	builder.addTensor("conv3_3_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv3_3_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv3_3", "conv3_out", "conv3_3_weight", "conv3_3_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv3_3_out");
					
	//layer: base_net.level2_0.eesp.spp_dw.0.conv.weight
	builder.addTensor("conv3_4_weight", {1, 3, 3, 16}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv3_4_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv3_4", "conv3_out", "conv3_4_weight", "conv3_4_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 2, 2, true, ANEURALNETWORKS_FUSED_NONE, "conv3_4_out");
    //name: 585, 587, 589                 
    builder.eltwise("add_1_layer3", "conv3_1_out", "conv3_2_out", ANEURALNETWORKS_FUSED_NONE, "add_1_layer3_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_2_layer3", "add_1_layer3_out", "conv3_3_out", ANEURALNETWORKS_FUSED_NONE, "add_2_layer3_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_3_layer3", "add_2_layer3_out", "conv3_4_out", ANEURALNETWORKS_FUSED_NONE, "add_3_layer3_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    
    //name: 627  
    builder.addTensor("pointwise3_1_weight", {64, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise3_1_bias", {64}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise3_1", "add_3_layer3_out", "pointwise3_1_weight", "pointwise3_1_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise3_1_out"
                   );
    //name: 629         
    builder.eltwise("add_pointwise3_4", "avg_pool4_out", "pointwise3_1_out", ANEURALNETWORKS_FUSED_NONE, 
                    "add_pointwise3_4_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);  
    builder.addTensor("pointwise3_6_weight", {128, 1, 1, 64}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise3_6_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise3_6", "add_pointwise3_4_out", "pointwise3_6_weight", "pointwise3_6_bias",           
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise3_6_out"
                   );
    //name: 631
    builder.avgpool("avg_pool5", "data", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 2, 2, 4, 4, 
	                ANEURALNETWORKS_FUSED_NONE, "avg_pool5_out");
    //name: 633
    builder.avgpool("avg_pool6", "avg_pool5_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 2, 2, 4, 4, 
	                ANEURALNETWORKS_FUSED_NONE, "avg_pool6_out");   
    //name: 635
    builder.avgpool("avg_pool7", "avg_pool6_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 2, 2, 4, 4, 
	                ANEURALNETWORKS_FUSED_NONE, "avg_pool7_out"); 
    //name: 636                
    builder.addTensor("conv3_5_weight", {3, 3, 3, 3}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv3_5_bias", {3}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv3_5", "avg_pool7_out", "conv3_5_weight", "conv3_5_bias",           
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv3_5_out");
    //name: 640
    builder.addTensor("conv3_6_weight", {128, 1, 1, 3}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv3_6_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv3_6", "conv3_5_out", "conv3_6_weight", "conv3_6_bias",           
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv3_6_out");
                   
    //name: 642         
    builder.eltwise("add_5_layer3", "pointwise3_6_out", "conv3_6_out", ANEURALNETWORKS_FUSED_RELU6, 
                    "add_5_layer3_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);

    /////////////////////// layer4
    //layer: base_net.level3.0.proj_1x1.conv.weight(645)
	builder.addTensor("conv4_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv4_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv4", "add_5_layer3_out", "conv4_weight", "conv4_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv4_out");
                    
	//layer: base_net.level3.0.spp_dw.0.conv.weight
	builder.addTensor("conv4_1_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv4_1_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv4_1", "conv4_out", "conv4_1_weight", "conv4_1_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv4_1_out");
					
	//layer: base_net.level3.0.spp_dw.1.conv.weight
	builder.addTensor("conv4_2_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv4_2_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv4_2", "conv4_out", "conv4_2_weight", "conv4_2_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv4_2_out");
					
	//layer: base_net.level3.0.spp_dw.2.conv.weight    
	builder.addTensor("conv4_3_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv4_3_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv4_3", "conv4_out", "conv4_3_weight", "conv4_3_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv4_3_out");
					
	//layer: base_net.level3.0.spp_dw.3.conv.weight
	builder.addTensor("conv4_4_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv4_4_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv4_4", "conv4_out", "conv4_4_weight", "conv4_4_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv4_4_out");
    //name: 651, 653, 655                 
    builder.eltwise("add_1_layer4", "conv4_1_out", "conv4_2_out", ANEURALNETWORKS_FUSED_NONE, "add_1_layer4_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_2_layer4", "add_1_layer4_out", "conv4_3_out", ANEURALNETWORKS_FUSED_NONE, "add_2_layer4_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_3_layer4", "add_2_layer4_out", "conv4_4_out", ANEURALNETWORKS_FUSED_RELU6, "add_3_layer4_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    
    //name: 660  
    builder.addTensor("pointwise4_1_weight", {128, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise4_1_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise4_1", "add_3_layer4_out", "pointwise4_1_weight", "pointwise4_1_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise4_1_out"
                   );
    
    //name: 662             
    builder.eltwise("add_4_layer4", "add_5_layer3_out", "pointwise4_1_out", ANEURALNETWORKS_FUSED_RELU6, "add_4_layer4_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
                 
    /////////////////////// layer5
    //layer: base_net.level3.1.proj_1x1.conv.weight(665)
	builder.addTensor("conv5_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv5_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv5", "add_4_layer4_out", "conv5_weight", "conv5_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv5_out");
                    
	//layer: base_net.level3.1.spp_dw.0.conv.weight(669)
	builder.addTensor("conv5_1_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv5_1_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv5_1", "conv5_out", "conv5_1_weight", "conv5_1_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv5_1_out");
					
	//layer: base_net.level3.1.spp_dw.1.conv.weight
	builder.addTensor("conv5_2_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv5_2_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv5_2", "conv5_out", "conv5_2_weight", "conv5_2_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv5_2_out");
					 
	//layer: base_net.level3.1.spp_dw.2.conv.weight 
	builder.addTensor("conv5_3_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv5_3_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv5_3", "conv5_out", "conv5_3_weight", "conv5_3_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv5_3_out");
					
	//layer: base_net.level3.1.spp_dw.3.conv.weight
	builder.addTensor("conv5_4_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv5_4_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv5_4", "conv5_out", "conv5_4_weight", "conv5_4_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv5_4_out");
    
    //name: 671, 673, 675                 
    builder.eltwise("add_1_layer5", "conv5_1_out", "conv5_2_out", ANEURALNETWORKS_FUSED_NONE, "add_1_layer5_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_2_layer5", "add_1_layer5_out", "conv5_3_out", ANEURALNETWORKS_FUSED_NONE, "add_2_layer5_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_3_layer5", "add_2_layer5_out", "conv5_4_out", ANEURALNETWORKS_FUSED_RELU6, "add_3_layer5_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    
    //name: 680  
    builder.addTensor("pointwise5_1_weight", {128, 1, 1, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise5_1_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise5_1", "add_3_layer5_out", "pointwise5_1_weight", "pointwise5_1_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise5_1_out"
                   );
    //name: 682             
    builder.eltwise("add_4_layer5", "add_4_layer4_out", "pointwise5_1_out", ANEURALNETWORKS_FUSED_RELU6, "add_4_layer5_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
                    
    /////////////////////// layer6
    //layer: base_net.level3.2.proj_1x1.conv.weight(685)
	builder.addTensor("conv6_weight", {32, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv6_bias", {32}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv6", "add_4_layer5_out", "conv6_weight", "conv6_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU6, "conv6_out");
                    
	//layer: base_net.level3.2.spp_dw.0.conv.weight(689)
	builder.addTensor("conv6_1_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv6_1_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv6_1", "conv6_out", "conv6_1_weight", "conv6_1_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv6_1_out");
					
	//layer: base_net.level3.2.spp_dw.1.conv.weight
	builder.addTensor("conv6_2_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv6_2_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv6_2", "conv6_out", "conv6_2_weight", "conv6_2_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv6_2_out");
					 
	//layer: base_net.level3.2.spp_dw.2.conv.weight 
	builder.addTensor("conv6_3_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv6_3_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv6_3", "conv6_out", "conv6_3_weight", "conv6_3_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                   1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv6_3_out");
					
	//layer: base_net.level3.2.spp_dw.3.conv.weight
	builder.addTensor("conv6_4_weight", {1, 3, 3, 32}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv6_4_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
	builder.conv2d("conv6_4", "conv6_out", "conv6_4_weight", "conv6_4_bias", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    1, 1, 1, 1, 1, 1, true, ANEURALNETWORKS_FUSED_NONE, "conv6_4_out");
    
    //name: 691, 693, 695                 
    builder.eltwise("add_1_layer6", "conv6_1_out", "conv6_2_out", ANEURALNETWORKS_FUSED_NONE, "add_1_layer6_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_2_layer6", "add_1_layer6_out", "conv6_3_out", ANEURALNETWORKS_FUSED_NONE, "add_2_layer6_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    builder.eltwise("add_3_layer6", "add_2_layer6_out", "conv6_4_out", ANEURALNETWORKS_FUSED_RELU6, "add_3_layer6_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
    
    //name: 700  
    builder.addTensor("pointwise6_1_weight", {128, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("pointwise6_1_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("pointwise6_1", "add_3_layer6_out", "pointwise6_1_weight", "pointwise6_1_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "pointwise6_1_out"
                   );
    //name: 704             
    builder.eltwise("add_4_layer6", "add_4_layer5_out", "pointwise6_1_out", ANEURALNETWORKS_FUSED_RELU6, "add_4_layer6_out",   
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, ELTWISE_ADDITION);
                    
    //name: 705  
    builder.addTensor("conv7_weight", {128, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv7_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv7", "add_4_layer6_out", "conv7_weight", "conv7_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv7_out"
                   );
    //name: 708
    builder.addTensor("conv8_weight", {128, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv8_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv8", "conv7_out", "conv8_weight", "conv8_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv8_out"
                   );               
    //name: 711
    builder.addTensor("conv9_weight", {128, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv9_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv9", "conv8_out", "conv9_weight", "conv9_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv9_out"
                   );                 
    //name: 714
    builder.addTensor("conv10_weight", {128, 3, 3, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv10_bias", {128}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv10", "conv9_out", "conv10_weight", "conv10_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 1, 1, 1, 1, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv10_out"
                   );
    //name: 719
    builder.addTensor("conv11_weight", {4, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv11_bias", {4}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv11", "conv10_out", "conv11_weight", "conv11_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_NONE, "conv11_out"
                   );
    //name: cls_score
    builder.addTensor("conv12_weight", {3, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv12_bias", {3}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv12", "conv10_out", "conv12_weight", "conv12_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv12_out"
                   );
    //name: centerness
    builder.addTensor("conv13_weight", {1, 1, 1, 128}, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, weightptr);
	builder.addTensor("conv13_bias", {1}, ANEURALNETWORKS_TENSOR_INT32, biasptr);
    builder.conv2d("conv13", "conv10_out", "conv13_weight", "conv13_bias", 
                   ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, 0, 0, 0, 0, 1, 1, false, ANEURALNETWORKS_FUSED_RELU, "conv13_out"
                   );

    // set input/output
    builder.setInputTensors("data", indataptr, ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("conv1_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("avg_pool1_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("conv2_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("conv2_1_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("conv2_2_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("conv2_3_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("conv2_4_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("pointwise2_1_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("pointwise2_2_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("pointwise2_3_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("pointwise2_4_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("add_pointwise2_1_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("add_pointwise2_2_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    // builder.setOutputTensors("add_pointwise2_3_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.setOutputTensors("conv11_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.setOutputTensors("conv12_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
    builder.setOutputTensors("conv13_out", ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);



    // compile
    builder.compile(deviceIndex);

    // warm up
    builder.execute();
    builder.execute();
    builder.execute();

    // execute
    double timesum = 0.0;
    const int EVALUATE_TIMES = 100;
    for (int t = 0; t < EVALUATE_TIMES; ++t)
    {
        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

        builder.execute();

        std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
        timesum += (std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count());
    }
    double fps = 1.0 / (timesum / EVALUATE_TIMES / 1000.0);
    SLOG_INFO << "FPS: " << fps << std::endl;

    // validate
    // std::vector<void *> results = builder.getOutput();

    // for (int32_t idx = 0; idx < 10; ++idx)
    // {
    //     uint8_t *result0 = reinterpret_cast<uint8_t *>(results[0]);
    //     SLOG_INFO << (int)(result0[idx]) << std::endl;
    // }

    delete [] indataptr;
    delete [] weightptr;
    delete [] biasptr;

    SLOG_INFO << "fine" << std::endl;
    return 0;
    //return fps;
}
