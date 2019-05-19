import torch
import h5py
import os
import numpy as np
import ipdb
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers

from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

from config.finetune import USE_GPU, GAUS_KERNEL, GAUS_STD

class DataSet(Dataset):
	def __init__(self, root_dir, image_transforms=None, mode="train", normalize=True, num_joints=17, heatmap2d=True):
		self.root_dir = root_dir
		self.train = mode.lower() == "train"
		self.image_transforms = image_transforms
		file_names = open(os.path.join(root_dir,"annot","%s_images.txt"%mode)).readlines()
		self.file_paths = [os.path.join(root_dir, "images", path[:-1]) for path in file_names]
		self.normalize = normalize
		self.heatmap2d = heatmap2d

		if self.train:
			annotations_path = os.path.join(root_dir,"annot","train.h5")
			target = h5py.File(annotations_path, 'r')
			self.target3d = torch.from_numpy(target['pose3d'][()].astype(np.float32))
			self.target2d = torch.from_numpy(target['pose2d'][()].astype(np.float32))
			if num_joints == 16:
				self.target2d = reduce_joints_to_16(self.target2d)

		self.mean = torch.from_numpy(np.loadtxt(os.path.join(root_dir,'annot',"mean.txt")).reshape([1, 17, 3]).astype(np.float32))
		self.std = torch.from_numpy(np.loadtxt(os.path.join(root_dir,'annot',"std.txt")).reshape([1, 17, 3]).astype(np.float32))

		if heatmap2d:
			self.gaussian_filter = GaussianSmoothing2D(num_joints, GAUS_KERNEL, GAUS_STD)

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, idx):
		image = io.imread(self.file_paths[idx]).astype(np.float32)
		image = np.moveaxis(image / 128.0 - 1, 2, 0)
		image = torch.from_numpy(image)
		if USE_GPU:
			image = image.cuda()
		if self.image_transforms:
			image = self.image_transforms(image)
		
		if not self.train:
			return image

		target_3D = self.target3d[idx]

		if self.normalize:
			target_3D = (target_3D - self.mean) / self.std

		sample = {'image': image, 'pose3d': target_3D.flatten()}

		if self.heatmap2d:
			heatmap = self.generate_2Dheatmaps(target_2D)
			sample['heatmap2d'] = heatmap

		return sample

	def generate_2Dheatmaps(self, joints, map_dim=64):
		"""
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
		for i, (p, q) in enumerate(zip(x, y)):
			maps[i, p/downscale, q/downscale] = 1
		maps = self.gaussian_filter(maps)
		return maps


class LiftingDataSet(Dataset):
	def __init__(self, root_dir, mode="train", normalize=True):
		self.root_dir = root_dir
		self.train = mode.lower() == "train"
		self.normalize = normalize
		if self.train:
			annotations_path = os.path.join(root_dir,"annot","train.h5")
			target = h5py.File(annotations_path, 'r')
			self.target3d = torch.from_numpy(target['pose3d'][()].astype(np.float32))
			self.target2d = torch.from_numpy(target['pose2d'][()].astype(np.float32))
			self.target2d = reduce_joints_to_16(self.target2d)
		else:
			# TODO: store outputs of 2D model
			pass

		self.mean = torch.from_numpy(np.loadtxt(os.path.join(root_dir,'annot',"mean.txt")).reshape([1, 17, 3]).astype(np.float32))
		self.std = torch.from_numpy(np.loadtxt(os.path.join(root_dir,'annot',"std.txt")).reshape([1, 17, 3]).astype(np.float32))

	def __getitem__(self, idx):
		target_2D = self.target2d[idx].flatten()
		if self.train:
			target_3D = self.target3d[idx]
			if self.normalize:
				target_3D = (target_3D - self.mean) / self.std
			sample = {'pose3d': target_3D.flatten(), 'pose2d': target_2D}
			return sample
		else:
			return target_2D

	def __len__(self):
		return self.target3d.shape[0]


def reduce_joints_to_16(joints_17):
	"""
	Maps 17 joints of H3.6M dataset to 16 joints of MPII
	H3.6 joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
	MPII joints = (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)

	Arguments:
		joints_17 (torch.Tensor): Tensor of shape (BATCH_SIZE, 17, 2) representing H3.6M coordinates.
	Returns: 
		joints_16 (torch.Tensor): Tensor of shape (BATCH_SIZE, 16, 2) representing MPII coordinates.
	"""
	permutation = [3, 2, 1, 4, 5, 6, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13]
	return joints_17[:, permutation, :]


class GaussianSmoothing2D(nn.Module):
	"""
	Arguments:
		channels (int): Number of channels of input. Output will have same number of channels.
		kernel_size (int): Size of the gaussian kernel.
		sigma (float): Standard deviation of the gaussian kernel.
		dim (int): Number of dimensions of the data.
		input_size (int): (H, W) Dimension of channel. Assumes H = W.
	"""
	def __init__(self, channels: int, kernel_size: int, sigma: float, dim: int = 2, input_size: int = 64):
		super(GaussianSmoothing2D, self).__init__()
		kernel_size = [kernel_size] * dim
		sigma = [sigma] * dim
		kernel = 1
		meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
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

	def forward(self, x):
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
		channel_norm = channel_norm.view(-1, 1).repeat(1, self.dim_input * self.dim_input)
		channel_norm = channel_norm.view(self.num_channels, self.dim_input, self.dim_input)
		return (x / channel_norm)