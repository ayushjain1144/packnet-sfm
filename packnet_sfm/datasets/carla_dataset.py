import glob
import numpy as np
import os

from torch.utils.data import Dataset

from packnet_sfm.geometry.pytorch_disco_utils import create_depth_image
from packnet_sfm.geometry.pytorch_disco_utils import scale_intrinsics, safe_inverse
from packnet_sfm.geometry.pose_utils import invert_pose_numpy

import torch
import pickle
from PIL import Image
from matplotlib import cm
import ipdb 
st = ipdb.set_trace

M = 8  # Image, around which context will be built
T = 19 # Total images in the video (stereo pairs for camera 0 - 9)
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

        backward_context = back_context
        assert backward_context >= 0 and forward_context >= 0, 'Invalid contexts'

        self.with_context = (backward_context != 0 or forward_context != 0) 
        
        self.back_context = back_context
        self.forward_context = forward_context
        self.strides = strides[0]
        # Obtaining the feed id
        self.split = file_list.split('/')[-1].split('.')[0]

        self.train = train
        self.root_dir = root_dir
        self.data_transform = data_transform

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.with_pose = with_pose

        self._cache = {}
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}


        # print(file_list)
        # print(root_dir)
        with open(os.path.join(root_dir, file_list), "r") as f:
            data = f.readlines()

        self.paths = []
        for i, fname in enumerate(data):
            # get file list
            path = os.path.join(root_dir, fname.split()[0])
            self.paths.append(path)

        # assert whether the context is within possible bounds

        assert(M - 2 * strides[0] * back_context >= 0)
        assert(M + 2 * strides[0] * forward_context <= T)

    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))
        return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../.."))

    ####################### Helper Functions ######################


    def __len__(self):
        return  len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index"""

        # loading feed dict
        
        feed = pickle.load(open(self.paths[idx], 'rb'))
        
        depth, _ = create_depth_image(torch.tensor(feed['pix_T_cams_raw'][M]).to(torch.float32).unsqueeze(0), torch.tensor(feed['xyz_camXs_raw'][M]).to(torch.float32).unsqueeze(0), 256, 256)

        #print(depth[0][0].shape)

        #print(feed['rgb_camXs_raw'][M])
        #import sys
        #sys.exit()
        sample = {
            'idx': idx,
            'filename': '%s_%010d' % (self.split, idx), 
            'rgb': Image.fromarray(feed['rgb_camXs_raw'][M]),
            'intrinsics': feed['pix_T_cams_raw'][M][:3, :3],
            'pose': feed['origin_T_camXs_raw'][M],
            'depth': depth[0][0].numpy(),
        }

        # Add context information

        if self.with_context:
            image_context = []
            for i in range(self.back_context):
                image_context.append(Image.fromarray(feed['rgb_camXs_raw'][M - 2 * self.strides * (i+1)]))
            for i in range(self.forward_context):
                image_context.append(Image.fromarray(feed['rgb_camXs_raw'][M + 2 * self.strides * (i+1)]))
            
            sample.update({
                'rgb_context': image_context
            })

            #Add context poses

            if self.with_pose:
                first_pose = sample['pose']
                image_context_pose = []

                for i in range(self.back_context):
                    image_context_pose.append(feed['origin_T_camXs_raw'][M - 2 * self.strides * (i+1)])
                for i in range(self.forward_context):
                    image_context_pose.append(feed['origin_T_camXs_raw'][M + 2 * self.strides * (i+1)])

                image_context_pose = [invert_pose_numpy(context_pose) @ first_pose
                                      for context_pose in image_context_pose]

                sample.update({
                    'pose_context': image_context_pose
                })


        if self.data_transform:
            sample = self.data_transform(sample)

        return sample


