import torch
import h5py
import os
import numpy as np

from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

from config import USE_GPU

class DataSet(Dataset):
	def __init__(self, root_dir, image_transforms=None, mode="train", normalize=True, num_joints=17):
		self.root_dir = root_dir
		self.train = mode.lower() == "train"
		self.image_transforms = image_transforms
		file_names = open(os.path.join(root_dir,"annot","%s_images.txt"%mode)).readlines()
		self.file_paths = [os.path.join(root_dir, "images", path[:-1]) for path in file_names]
		self.normalize = normalize

		if self.train:
			annotations_path = os.path.join(root_dir,"annot","train.h5")
			target = h5py.File(annotations_path, 'r')
			self.target3d = torch.from_numpy(target['pose3d'][()].astype(np.float32))
			self.target2d = torch.from_numpy(target['pose2d'][()].astype(np.float32))
			if num_joints == 16:
				self.target2d = reduce_joints_to_16(self.target2d)

		self.mean = torch.from_numpy(np.loadtxt(os.path.join(root_dir,'annot',"mean.txt")).reshape([1, 17, 3]).astype(np.float32))
		self.std = torch.from_numpy(np.loadtxt(os.path.join(root_dir,'annot',"std.txt")).reshape([1, 17, 3]).astype(np.float32))


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
		target_2D = self.target2d[idx].flatten()

		if self.normalize:
			target_3D = (target_3D - self.mean) / self.std

		sample = {'image': image, 'pose3d': target_3D.flatten(), 'pose2d': target_2D}
		return sample

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
	H3.6 joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
	MPII joints = (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)

	args: joints_17 -- type(H3.6), shape: (17, 2)
	return: joints_16 -- type(MPII), shape: (16, 2)
	"""
	permutation = [3, 2, 1, 4, 5, 6, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13]
	return joints_17[:, permutation, :]
