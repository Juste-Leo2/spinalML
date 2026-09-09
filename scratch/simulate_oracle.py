import numpy as np

# Test vectors from test_vectors.json
in_ints = [-4, 3, -5, 2, -6, 1, -7, 0, 7, -1, 6, -2, 5, -3, 4, -4]
x = np.array(in_ints, dtype=np.int16).reshape(8, 2)
print("Input shape:", x.shape)
print("Input:\n", x)

# Conv1D weights: kernelSize=3, inChannels=2, outChannels=2
# Total elements = 3 * 2 * 2 = 12
# In WeightMemoryLayout: (idx % 7) + 1 -> 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5
w_conv = np.array([((i % 7) + 1) for i in range(12)], dtype=np.int16).reshape(3, 2, 2)
b_conv = np.array([((i % 5) + 1) for i in range(2)], dtype=np.int16)
print("Conv1D weights shape:", w_conv.shape, "values:", w_conv.flatten())
print("Conv1D bias:", b_conv)

# Conv1D: out_l = 8 - 3 + 1 = 6
# For each step l in 0..5, out[l, c_out] = sum_{k=0..2} sum_{c_in=0..1} x[l+k, c_in] * w[k, c_in, c_out] + b[c_out]
conv_out = np.zeros((6, 2), dtype=np.int32)
for l in range(6):
    for co in range(2):
        acc = b_conv[co]
        for k in range(3):
            for ci in range(2):
                acc += x[l+k, ci] * w_conv[k, ci, co]
        conv_out[l, co] = acc

print("Conv1D output (6, 2):\n", conv_out)

# AvgPool1D(poolSize = 2, stride = 2)
# out shape: 6 / 2 = 3, 2
pool_out = np.zeros((3, 2), dtype=np.int32)
for l in range(3):
    for c in range(2):
        pool_out[l, c] = (conv_out[2*l, c] + conv_out[2*l+1, c]) // 2

print("AvgPool1D output (3, 2):\n", pool_out)

# ReLU
relu_out = np.maximum(0, pool_out)
print("ReLU output:\n", relu_out)

# Flatten -> (1, 6)
flat = relu_out.flatten()
print("Flatten output (6,):\n", flat)

# Linear(inFeatures = 6, outFeatures = 2)
# w_linear: 6 * 2 = 12 elements: ((i % 7) + 1)
w_lin = np.array([((i % 7) + 1) for i in range(12)], dtype=np.int32).reshape(6, 2)
b_lin = np.array([((i % 5) + 1) for i in range(2)], dtype=np.int32)
print("Linear weights (6, 2):\n", w_lin)
print("Linear bias:", b_lin)

lin_out = np.dot(flat, w_lin) + b_lin
print("Linear output:", lin_out)

# Requantize(shift = 1, targetType = I8)
shifted = lin_out >> 1
sat = np.clip(shifted, -128, 127)
print("Requantize shifted:", shifted)
print("Requantize saturated (I8):", sat)
