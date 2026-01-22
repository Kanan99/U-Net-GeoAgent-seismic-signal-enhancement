import numpy as np

def overlap_average(patches, positions, shape, patch_size):
    output = np.zeros(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)

    for patch, (t, x) in zip(patches, positions):
        output[t:t+patch_size, x:x+patch_size] += patch
        weight[t:t+patch_size, x:x+patch_size] += 1

    return output / np.maximum(weight, 1e-6)
