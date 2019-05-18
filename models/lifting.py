import ipdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def weight_init(l):
	if isinstance(l, nn.Linear):
		nn.init.kaiming_normal_(l.weight)

class MartinezLinear(nn.Module):
	def __init__(self, lsize: int, dropout: float):
		super(MartinezLinear, self).__init__()
		self.lsize = lsize
		self.dropout = nn.Dropout(dropout)
		self.branch_1 = nn.Linear(self.lsize, self.lsize)
		self.bn_1 = nn.BatchNorm1d(self.lsize)
		self.branch_2 = nn.Linear(self.lsize, self.lsize)
		self.bn_2 = nn.BatchNorm1d(self.lsize)

	def forward(self, x):
		y = self.branch_1(x)
		y = self.bn_1(y)
		y = F.relu(y)
		y = self.dropout(y)
		y = self.branch_2(y)
		y = self.bn_2(y)
		y = F.relu(y)
		y = self.dropout(y)

		return (x+y)

class MartinezModel(nn.Module):
	def __init__(self, lsize=1024, nblocks=2, p=0.5, input_size=16*2, output_size=17*3):
		super(MartinezModel, self).__init__()
		self.lsize = lsize
		self.nblocks = nblocks
		self.dropout = nn.Dropout(p)
		self.input_size = input_size
		self.output_size = output_size
		self.l_1 = nn.Linear(self.input_size, self.lsize)
		self.bn_1 = nn.BatchNorm1d(self.lsize)
		self.linear_blocks = nn.ModuleList([MartinezLinear(self.lsize, p) for _ in range(nblocks)])
		self.l_2 = nn.Linear(self.lsize, self.output_size)

	def forward(self, x):
		y = self.l_1(x)
		y = self.bn_1(y)
		y = F.relu(y)
		y = self.dropout(y)
		for i in range(self.nblocks):
			y = self.linear_blocks[i](y)
		y = self.l_2(y)
		return y

class CyclicalMartinez(nn.Module):
	""" A proposed model that re-projects 3D coordinates to 2D plane"""
	def __init__(self, cfg):
		super(CyclicalMartinez, self).__init__()
		self.inward = MartinezModel(cfg.TWOD['LINEAR_SIZE'], cfg.TWOD['NUM_BLOCKS'], cfg.TWOD['p'])
		self.outward = MartinezModel(cfg.THREED['LINEAR_SIZE'], cfg.THREED['NUM_BLOCKS'], cfg.THREED['p'], input_size=17*3, output_size=16*2)

	def forward(self, x):
		y3d = self.inward(x)
		y2d = self.outward(y3d)

		return {'pose_3d': y3d, 'pose_2d': y2d}