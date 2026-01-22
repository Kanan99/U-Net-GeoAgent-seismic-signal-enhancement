import numpy as np

def extract_patches(section, patch_size, stride):
    patches = []
    positions = []
    H, W = section.shape

    for t in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patches.append(section[t:t+patch_size, x:x+patch_size])
            positions.append((t, x))

    return np.array(patches), positions
