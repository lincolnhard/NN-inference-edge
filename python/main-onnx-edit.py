import onnx
from onnx import optimizer

# Preprocessing: load the model to be optimized.
model_path = 'espnetv2fusion.onnx'
original_model = onnx.load(model_path)

all_passes = optimizer.get_available_passes()
# print("Available optimization passes:")
# for p in all_passes:
#     print(p)
# print()

passes = ['fuse_consecutive_transposes',
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_dropout',
        'eliminate_nop_monotone_argmax',
        'eliminate_nop_pad',
        'eliminate_nop_transpose',
        'eliminate_unused_initializer',
        'extract_constant_to_initializer',
        'fuse_add_bias_into_conv',
        # 'fuse_bn_into_conv',
        'fuse_consecutive_concats',
        'fuse_consecutive_log_softmax',
        'fuse_consecutive_reduce_unsqueeze',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
        'fuse_pad_into_conv',
        # 'fuse_transpose_into_gemm',
        # 'lift_lexical_references',
        'nop',
        # 'split_init',
        # 'split_predict'
        ]

# passes = ['fuse_consecutive_transposes',
#         'fuse_bn_into_conv'
#         ]

# Apply the optimization on the original model
optimized_model = optimizer.optimize(original_model, passes)

onnx.checker.check_model(optimized_model)

onnx.save(optimized_model, 'espnetv2fusion_optimized.onnx')
