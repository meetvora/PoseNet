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

logFormatter = "%(levelname)s: %(message)s"
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_new_HR2D():
	""" 
	Returns an instance of HRNet with new final Conv2d layer
	"""
	model = models.hrnet.PoseHighResolutionNet(config.hrnet)
	model.init_weights(config.hrnet.PRETRAINED, config.finetune.USE_GPU)
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
	normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_set = DataSet(config.finetune.DATA_PATH, image_transforms=normalize, num_joints=17)
	train_loader = DataLoader(train_set, batch_size=config.finetune.BATCH_SIZE, num_workers=config.finetune.WORKERS, shuffle=True)

	model = get_new_HR2D()
	print_all_attr(config.finetune)
	train(model, train_loader)
	logger.info("[+] Finished training.\nSaving model...")
	torch.save(model.state_dict(), os.path.join(config.finetune.LOG_PATH, config.finetune.NAME))
	logger.info("[+] Saved final model at %s" % os.path.join(config.finetune.LOG_PATH, config.finetune.NAME))

if __name__ == '__main__':
	main()