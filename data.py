import torch
import h5py
import os
import numpy as np

from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

class DataSet(Dataset):
	def __init__(self, root_dir, image_transforms=None, mode="train", normalize=True):
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

		self.mean = torch.from_numpy(np.loadtxt(os.path.join(root_dir,'annot',"mean.txt")).reshape([1, 17, 3]).astype(np.float32))
		self.std = torch.from_numpy(np.loadtxt(os.path.join(root_dir,'annot',"std.txt")).reshape([1, 17, 3]).astype(np.float32))


	def __len__(self):
		return len(self.file_paths)


	def __getitem__(self, idx):
		image = io.imread(self.file_paths[idx]).astype(np.float32)
		image = np.moveaxis(image / 128.0 - 1, 2, 0)
		image = torch.from_numpy(image)
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
