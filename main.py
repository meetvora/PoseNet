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
import models.hrnet
import models.posenet
import config
import config.hrnet
import config.posenet
from data import DataSet
from utils import *

if config.USE_GPU:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

logFormatter = "%(asctime)s - [%(levelname)s] %(message)s"
logging.basicConfig(filename=config.LOG_NAME, filemode='a', format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def train(model, train_loader):
	model.train()
	optimizer = getattr(optim, config.OPTIMIZER)(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
	overall_iter = 0
	JointLoss = JointsMSELoss()
	logger.info("[+] Starting training.")

	for epoch in range(config.NUM_EPOCHS):
		for batch_idx, sample in enumerate(train_loader):
			image, pose3d, heatmap2d = sample['image'], sample['pose3d'], sample['heatmap2d']
			if config.USE_GPU:
				image, pose3d, heatmap2d = to_cuda([image, pose3d, heatmap2d])
			optimizer.zero_grad()
			output = model(image)

			termwise_loss = {
				'heatmap': JointLoss(output['hrnet_maps'], heatmap2d),
				'cyclic_inward': F.mse_loss(output['cycl_martinez']['pose_3d'], pose3d),
				'cyclic_outward': F.mse_loss(output['cycl_martinez']['pose_2d'], output['hrnet_coord'])
			}

			loss = config.posenet.LOSS_COEFF['hrnet_maps'] * termwise_loss['heatmap'] + \
				config.posenet.LOSS_COEFF['cycl_martinez']['pose_3d'] * termwise_loss['cyclic_inward'] + \
				config.posenet.LOSS_COEFF['cycl_martinez']['pose_2d'] * termwise_loss['cyclic_outward']

			loss.backward()
			optimizer.step()

			if batch_idx % 10 == 0:
				mpjpe = compute_MPJPE(output['cycl_martinez']['pose_3d'].detach(), pose3d.detach(), train_loader.dataset.std.numpy())
				logger.debug(f'Train Epoch: {epoch} [{batch_idx}]\tTotal Loss: {loss.item():.6f}\tMPJPE: {mpjpe:.6f}')
				logger.debug(print_termwise_loss(termwise_loss))

			overall_iter += 1
			if overall_iter % config.SAVE_ITER_FREQ == 0:
				torch.save(model.state_dict(), os.path.join(config.LOG_PATH, config.NAME))

def evaluate(model, eval_loader, pretrained=False):
	if pretrained:
		model.load_state_dict(torch.load(os.path.join(config.LOG_PATH, config.NAME)))

	logger.info("[+] Starting evaluation.")
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
	generate_submission(prediction, "submission-%s.csv.gz"%(config.NAME))
	create_zip_code_files("code-%s.zip"%(config.NAME))

def main():
	normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_set = DataSet(config.DATA_PATH, image_transforms=normalize, num_joints=17)
	train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS, shuffle=True)

	eval_set = DataSet(config.DATA_PATH, normalize=False, mode="valid", image_transforms=normalize, heatmap2d=False)
	eval_loader = DataLoader(eval_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)

	model = models.posenet.PoseNet(config.posenet)

	print_all_attr([config, config.posenet], logger)

	train(model, train_loader)
	evaluate(model, eval_loader, pretrained=False)

if __name__ == '__main__':
	try:
		main()
	except Exception as e:
		logger.error("[!] Exception occurred.", exc_info=True)
