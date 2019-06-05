import torch
import h5py
import os
import numpy as np
import ipdb
import torch.nn as nn
import torch.nn.functional as F
import math

from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple

from config import USE_GPU
from config.finetune import GAUS_KERNEL, GAUS_STD


class DataSet(Dataset):
    def __init__(self,
                 root_dir: str,
                 image_transforms: List[str] = [],
                 transform_params: List[Tuple] = [],
                 mode: str = "train",
                 num_joints: int = 17,
                 heatmap2d: bool = True) -> None:
        self.root_dir = root_dir
        if mode not in ["train", "valid"]:
            raise ValueError("Unsupported mode.")
        self.train = mode.lower() == "train"
        file_names = open(
            os.path.join(root_dir, "annot",
                         "%s_images.txt" % mode)).readlines()
        self.file_paths = [
            os.path.join(root_dir, "images", path[:-1]) for path in file_names
        ]
        self.heatmap2d = heatmap2d
        self.image_preprocess, self.image_transforms = [], []
        self.transforms_ops = image_transforms

        if self.train:
            annotations_path = os.path.join(root_dir, "annot", "train.h5")
            target = h5py.File(annotations_path, 'r')
            self.target3d = torch.from_numpy(target['pose3d'][()].astype(
                np.float32))
            self.target2d = torch.from_numpy(target['pose2d'][()].astype(
                np.float32))
            if num_joints == 16:
                self.target2d = self._reduce_joints_to_16(self.target2d)
            self.image_preprocess.append(
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2))

        self.mean = torch.from_numpy(
            np.loadtxt(os.path.join(root_dir, 'annot', "mean.txt")).reshape(
                [1, 17, 3]).astype(np.float32))
        self.std = torch.from_numpy(
            np.loadtxt(os.path.join(root_dir, 'annot',
                                    "std.txt")).reshape([1, 17, 3
                                                         ]).astype(np.float32))

        self.image_preprocess = transforms.Compose(self.image_preprocess)
        self.img_mean = torch.as_tensor((0.485, 0.456, 0.406),
                                        dtype=torch.float32,
                                        device="cpu")
        self.img_std = torch.as_tensor((0.229, 0.224, 0.225),
                                       dtype=torch.float32,
                                       device="cpu")

        if heatmap2d:
            self.gaussian_filter = GaussianSmoothing2D(num_joints, GAUS_KERNEL,
                                                       GAUS_STD)

        if image_transforms:
            assert len(image_transforms) == len(transform_params)
            self.image_transforms = [
                getattr(transforms, op)(*param)
                for op, param in zip(self.transforms_ops, transform_params)
            ]
            self.joint_transforms = [
                getattr(self, op) for op in self.transforms_ops
            ]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.file_paths[idx])
        image = self.image_preprocess(image)

        if self.train:
            target_3D = (self.target3d[idx] - self.mean) / self.std
            target_2D = self.target2d[idx]
            p = np.random.rand(1)

            if self.image_transforms and p >= 0.5:
                for idx_ in range(len(self.image_transforms)):
                    image = self.image_transforms[idx_](image)
                    target_2D = self.joint_transforms[idx_](target_2D)

        image = transforms.functional.to_tensor(image)
        image = image.sub_(self.img_mean[:, None, None]).div_(
            self.img_std[:, None, None])
        image = image.cuda() if USE_GPU else image

        if not self.train:
            return image

        sample = {
            'image': image,
            'pose3d': target_3D.flatten(),
            'pose2d': target_2D.flatten()
        }

        if self.heatmap2d:
            heatmap = self._generate_2Dheatmaps(target_2D)
            sample['heatmap2d'] = heatmap

        return sample

    def _generate_2Dheatmaps(self, joints: torch.Tensor,
                             map_dim: int = 64) -> torch.Tensor:
        """ Generates 2d heatmaps from coordinates. 
		Arguments:
			joints (torch.Tensor): An individual target tensor of shape (num_joints, 2).
		Returns:
			maps (torch.Tensor): 3D Tensor with gaussian activation at joint locations (num_joints, map_dim, map_dim)
		"""
        joints = joints.to(dtype=torch.int16)
        num_joints = joints.shape[0]
        downscale = 256 / map_dim
        maps = torch.zeros((num_joints, map_dim, map_dim))
        x, y = joints[:, 0].long(), joints[:, 1].long()
        maps[np.arange(num_joints), x // downscale, y // downscale] = 1
        maps = self.gaussian_filter(maps)
        return maps

    def _reduce_joints(self, joints_17: torch.Tensor) -> torch.Tensor:
        """
		Maps 17 joints of H3.6M dataset to 16 joints of MPII
		H3.6 joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
		MPII joints = (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)

		Arguments:
			joints_17 (torch.Tensor): Tensor of shape (BATCH_SIZE, 17, 2) representing H3.6M coordinates.
		Returns:
			joints_16 (torch.Tensor): Tensor of shape (BATCH_SIZE, 16, 2) representing MPII coordinates.
		"""
        assert joints_17.shape[1:] == (17, 2)

        permutation = [3, 2, 1, 4, 5, 6, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13]
        return joints_17[:, permutation, :]

    def RandomHorizontalFlip(self, joints: torch.Tensor) -> torch.Tensor:
        pairs = np.asarray([[1, 4], [2, 5], [3, 6], [14, 11], [15, 12],
                            [16, 13]])
        joints[pairs, :] = torch.flip(joints[pairs, :], (1, ))
        return joints


class GaussianSmoothing2D(nn.Module):
    """
	Arguments:
		channels (int): Number of channels of input. Output will have same number of channels.
		kernel_size (int): Size of the gaussian kernel.
		sigma (float): Standard deviation of the gaussian kernel.
		dim (int): Number of dimensions of the data.
		input_size (int): (H, W) Dimension of channel. Assumes H = W.
	"""

    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 sigma: float,
                 dim: int = 2,
                 input_size: int = 64) -> None:
        super(GaussianSmoothing2D, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
              torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.num_channels = channels
        self.dim_input = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
		Apply gaussian filter to input.
		Arguments:
			input (torch.Tensor): Input of shape (C, H, W) to apply gaussian filter on.
		Returns:
			filtered (torch.Tensor): Filtered output of same shape.
		"""
        x = F.pad(x, (1, 1, 1, 1))
        x = x.unsqueeze(0).float()
        x = F.conv2d(x, weight=self.weight, groups=self.groups).squeeze()
        channel_norm = torch.norm(x.view(self.num_channels, -1), 2, 1)
        channel_norm = channel_norm.view(-1, 1).repeat(
            1, self.dim_input * self.dim_input)
        channel_norm = channel_norm.view(self.num_channels, self.dim_input,
                                         self.dim_input)
        return (x / channel_norm)
