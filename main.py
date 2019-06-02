import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.data import DataLoader

import models
import models.posenet
import config
import config.posenet
from data import DataSet
from utils import *

if config.USE_GPU:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

logging.basicConfig(**config.__LOG_PARAMS__)
logger = logging.getLogger(__name__)

def train(model, train_loader, eval_loader):
	"""
	Train a PoseNet model given parameters in config
	Arguments:
		model (nn.Module) - PoseNet instance
		train_loader (torch.data.DataLoader) - Dataloader for training data
		evak_loader (torch.data.DataLoader) - Dataloader for validation data
	"""
	optimizer = getattr(optim, config.OPTIMIZER)(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
	overall_iter = 0
	JointLoss = JointsMSELoss()
	logger.info("[+] Starting training.")

	for epoch in range(config.NUM_EPOCHS):
		model.train()
		for batch_idx, sample in enumerate(train_loader):
			image, pose3d, heatmap2d = sample['image'], sample['pose3d'], sample['heatmap2d']
			if config.USE_GPU:
				image, pose3d, heatmap2d = to_cuda([image, pose3d, heatmap2d])
			optimizer.zero_grad()
			output = model(image)

			termwise_loss = {
				'heatmap': JointLoss(output['hrnet_maps'], heatmap2d),
				'cyclic_inward': F.mse_loss(output['cycl_martinez']['pose_3d'], pose3d),
			}

			loss = config.posenet.LOSS_COEFF['hrnet_maps'] * termwise_loss['heatmap'] + \
				config.posenet.LOSS_COEFF['cycl_martinez']['pose_3d'] * termwise_loss['cyclic_inward']

			loss.backward()
			optimizer.step()

			if batch_idx % config.PRINT_BATCH_FREQ == 0:
				mpjpe = compute_MPJPE(output['cycl_martinez']['pose_3d'].detach(), pose3d.detach(), train_loader.dataset.std.numpy())
				logger.debug(f'Train Epoch: {epoch} [{batch_idx}]\tTotal Loss: {loss.item():.6f}\tMPJPE: {mpjpe:.6f}')
				logger.debug(print_termwise_loss(termwise_loss))

			overall_iter += 1
			if overall_iter % config.SAVE_ITER_FREQ == 0:
				torch.save(model.state_dict(), os.path.join(config.LOG_PATH, config.NAME + f"-iter={overall_iter}"))

		evaluate(model, eval_loader, epoch)

def evaluate(model, eval_loader, epoch, pretrained=False):
	if pretrained:
		model.load_state_dict(torch.load(os.path.join(config.LOG_PATH, config.NAME)))

	logger.info("[+] Evaluating at end of epoch %s." % epoch)
	with torch.no_grad():
		model.eval()
		prediction = list()

		for batch_idx, sample in enumerate(eval_loader):
			image = sample
			output = model(image)
			p3d_out = output['cycl_martinez']['pose_3d']
			p3d_out = unnormalize_pose(p3d_out, eval_loader.dataset.mean, eval_loader.dataset.std).astype(np.int16)
			prediction = np.append(prediction, p3d_out)

	prediction = prediction.reshape(-1, 51)
	generate_submission(prediction, "submission-%s-epoch-%s.csv.gz"%(config.NAME, epoch))

def main():
	normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_set = DataSet(config.DATA_PATH, mode="train")
	train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)

	eval_set = DataSet(config.DATA_PATH, mode="valid", heatmap2d=False)
	eval_loader = DataLoader(eval_set, batch_size=config.BATCH_SIZE)

	model = models.posenet.PoseNet(config.posenet)

	print_all_attr([config, config.posenet], logger)

	train(model, train_loader, eval_loader)

	create_zip_code_files("code-%s.zip"%(config.NAME))

if __name__ == '__main__':
	try:
		main()
	except Exception as e:
		logger.error("[!] Exception occurred.", exc_info=True)
