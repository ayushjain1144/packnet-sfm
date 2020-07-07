import glob
import numpy as np
import os

from torch.utils.data import Dataset

from packnet_sfm.geometry.pytorch_disco_utils import create_depth_image

def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)
    assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)


class CARLADataset(Dataset):
    
    def __init__(self, root_dir, file_list, train=True,
        data_transform=None, depth_type=None, with_pose=False,
        back_context=0, forward_context=0, strides=(1,)):
        # Assertions