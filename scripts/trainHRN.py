import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import logging
import sys
import ipdb
import models
import models.hrnet
import config.finetune
import config.hrnet
import data
from data import DataSet
from utils import *

if config.finetune.USE_GPU:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

logFormatter = "%(asctime)s - [%(levelname)s] %(message)s"
logging.basicConfig(filename=config.finetune.LOG_NAME, filemode='a', format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_new_HR2D():
	""" 
	Returns an instance of HRNet with new final Conv2d layer
	"""
	model = models.hrnet.PoseHighResolutionNet(config.hrnet)
	model.init_weights(None, config.finetune.USE_GPU)
	final_layer = nn.Conv2d(32, 17, kernel_size=(1, 1), stride=(1, 1))
	nn.init.normal_(final_layer.weight, std=0.001)
	for name, _ in final_layer.named_parameters():
		if name in ['bias']:
			nn.init.constant_(final_layer.bias, 0)
	model.final_layer = final_layer
	for param in model.parameters():
		param.requires_grad = True
	return model

def train(model, train_loader):
	model.train()
	optimizer = getattr(optim, config.finetune.OPTIMIZER)(model.parameters(), lr=config.finetune.LEARNING_RATE, weight_decay=config.finetune.WEIGHT_DECAY)
	overall_iter = 0
	JointLoss = JointsMSELoss()

	logger.info("[+] Starting training.")
	for epoch in range(config.finetune.NUM_EPOCHS):
		for batch_idx, sample in enumerate(train_loader):
			image, heatmap2d = sample['image'], sample['heatmap2d']
			if config.finetune.USE_GPU:
				image, heatmap2d = to_cuda([image, heatmap2d])
			optimizer.zero_grad()
			output = model(image)
			loss = JointLoss(output, heatmap2d)
			loss.backward()
			optimizer.step()

			if batch_idx % 500 == 0:
				logger.debug(f'Train Epoch: {epoch} [{batch_idx}]\tLoss: {loss.item():.6f}')

			overall_iter += 1
			if overall_iter % config.finetune.SAVE_ITER_FREQ == 0:
				torch.save(model.state_dict(), os.path.join(config.finetune.LOG_PATH, config.finetune.NAME))

def main():
	train_set = DataSet(config.DATA_PATH, mode="train", image_transforms=["RandomHorizontalFlip"], transform_params=[(1, )])
	train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)

	model = get_new_HR2D()
	print_all_attr([config, config.finetune], logger)
	train(model, train_loader)
	logger.info("[+] Finished training.\nSaving model...")
	torch.save(model.state_dict(), os.path.join(config.finetune.LOG_PATH, config.finetune.NAME))
	logger.info("[+] Saved final model at %s" % os.path.join(config.finetune.LOG_PATH, config.finetune.NAME))

if __name__ == '__main__':
	main()
