import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from models.lifting import *
from models.hrnet import *

class CyclicalMartinez(nn.Module):
	"""
	Inspired by Martinez et al.
	Regresses from 2D to 3D and back to 2D.
	Input: Batch of coordinates
	Output: Batch of 2D and 3D coorindates.
	"""
	def __init__(self, cfg):
		super(CyclicalMartinez, self).__init__()
		self.inward = MartinezModel(cfg.TWOD['LINEAR_SIZE'], cfg.TWOD['NUM_BLOCKS'], cfg.TWOD['p'], \
			input_size=cfg.TWOD['IN_SIZE'], output_size=cfg.TWOD['OUT_SIZE'])
		self.outward = MartinezModel(cfg.THREED['LINEAR_SIZE'], cfg.THREED['NUM_BLOCKS'], cfg.THREED['p'], \
			input_size=cfg.THREED['IN_SIZE'], output_size=cfg.THREED['OUT_SIZE'])

	def forward(self, x):
		y3d = self.inward(x)
		y2d = self.outward(y3d)
		return {'pose_3d': y3d, 'pose_2d': y2d}


class Argmax(nn.Module):
	"""
	Module to extract coordinates from heatmaps.
	Can switch between soft (differentiable) and hard (non-differentiable)
	"""
	def __init__(self, SOFTARGMAX):
		super(Argmax, self).__init__()
		self.get_coordinates = self.softargmax if SOFTARGMAX else self.hardargmax

	def hardargmax(self, maps):
		"""
		Converts 2D Heatmaps to coordinates.
		(NOTE: Recheck the mapping function and rescaling heuristic.)
		Arguments:
		    maps (torch.Tensor): 2D Heatmaps of shape (BATCH_SIZE, num_joins, 64, 64)
		Returns:
		    z (torch.Tensor): Coordinates of shape (BATCH_SIZE, num_joints*2)
		"""
		_, idx = torch.max(maps.flatten(2), 2)
		x, y = idx / 16  + 2, torch.remainder(idx, 64) * 4 + 2 # Rescaling to (256, 256)
		z = torch.stack((x, y), 2).flatten(1).float()
		return z

	def softargmax(self, maps, beta: float = 1e7, dim: int = 64):
		"""
		Applies softargmax to heatmaps and returns 2D (x,y) coordinates
		Arguments:
			maps (torch.Tensor): 2D Heatmaps of shape (BATCH_SIZE, num_joint, dim, dim)
			beta (float): Exponentiating constant. Default = 100000
			dim (int): Spatial dimension of map. Default = 64
		Returns:
			# values (torch.Tensor): max value of heatmap; shape (BATCH_SIZE, num_joints)
			regress_coord (torch.Tensor): (x, y) co-ordinates of shape (BATCH_SIZE, num_joints*2)
		"""
		batch_size, num_joints = maps.shape[0], maps.shape[1]
		flat_map = maps.view(batch_size, num_joints, -1).float()
		Softmax = nn.Softmax(dim=-1)
		softmax = Softmax(flat_map * beta).float()
		# values = torch.sum(flat_map * softmax, -1)
		posn = torch.arange(0, dim * dim).repeat(batch_size, num_joints, 1)
		idxs = torch.sum(softmax * posn.float(), -1).int()
		x, y = (idxs/dim)*4 + 2, torch.remainder(idxs, dim) * 4 + 2
		regress_coord = torch.stack((x, y), 2).float()
		regress_coord = regress_coord.flatten(1)
		return regress_coord

	def forward(self, x):
		return self.get_coordinates(x)


class PoseNet(nn.Module):
	"""
	The final supremo model that consists of two sub-models
		twoDNet:
			Pose HighResolution Net, proposed by Sun et al.
			Frozen and uses publicly available pretrained weights.
			Outputs: heatmap of shape (BATCH_SIZE, 16, 64, 64)
		liftNet:
			A 2D to 3D regressor and back to 2D.
			Basic block inspired by Martinez et al.
			Uses `map_to_coord` to convert heatmaps to 2D coordinates.
			Outputs: 3D coordinates and reprojected 2D coordinates.
	Input: Batch of images
	Output: 2D heatmaps, 2D coords by twoDNet, 3D and 2D coords by liftNet.
	NOTE:
		The outputs of twoDNet pass though Argmax to extract coordinates
			for liffNet
		We use pretrained weights for twoDNet but train the entire model
			in an end-to-end fashion.
		Pass config.posenet during initialization
	"""
	def __init__(self, cfg):
		super(PoseNet, self).__init__()
		self.twoDNet = PoseHighResolutionNet(cfg)
		# Update NUM_JOINTS in config.posenet to include new/prev final_layer
		# self.twoDNet.final_layer = nn.Conv2d(32, 17, kernel_size=(1, 1), stride=(1, 1))
		self.twoDNet.init_weights(cfg.PRETRAINED, cfg.USE_GPU)
		for param in self.twoDNet.parameters():
			param.requires_grad = cfg.END_TO_END
		self.liftNet = CyclicalMartinez(cfg)
		self.liftNet.apply(weight_init)
		self.argmax = Argmax(cfg.SOFTARGMAX)

	def forward(self, x):
		twoDMaps = self.twoDNet(x)
		twoDCoords = self.argmax(twoDMaps)
		liftOut = self.liftNet(twoDCoords)
		return {'hrnet_coord': twoDCoords, 'cycl_martinez': liftOut, 'hrnet_maps': twoDMaps}
