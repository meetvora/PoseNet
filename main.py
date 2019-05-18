import os
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.data import DataLoader

import models
import models.hrnet
import config
import config.hrnet
from data import DataSet, LiftingDataSet, reduce_joints_to_16
from utils import *

if config.USE_GPU:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_model(model, train_loader):
	model.train()
	optimizer = getattr(optim, config.OPTIMIZER)(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
	overall_iter = 0

	print("[+] Starting training.")
	for epoch in range(config.NUM_EPOCHS):
		for batch_idx, sample in enumerate(train_loader):
			image, pose2d, pose3d = sample['image'], sample['pose2d'], sample['pose3d']
			if config.USE_GPU:
				image, pose2d, pose3d = to_cuda([image, pose2d, pose3d])
			optimizer.zero_grad()
			# noise = torch.from_numpy(np.random.normal(scale=config.NOISE_STD, size=pose2d.shape).astype(np.float32))
			# inp = pose2d + noise
			output = model(image)
			loss = config.CYCLICAL_LOSS_COEFF[0] * F.mse_loss(output['cycl_martinez']['pose_3d'], pose3d) \
					+ config.CYCLICAL_LOSS_COEFF[1] * F.mse_loss(output['cycl_martinez']['pose_2d'], output['hrnet_coord'])
			loss.backward()
			optimizer.step()

			if batch_idx % 1 == 0:
				mpjpe, mpjpe_std = compute_MPJPE(output['cycl_martinez']['pose_3d'].detach(), pose3d.detach(), train_loader.dataset.std.numpy())
				hrnet_loss = F.mse_loss(output['hrnet_coord'], pose2d)
				print(f'Train Epoch: {epoch} [{batch_idx}]\tLoss: {loss.item():.6f}\tMPJPE: {mpjpe:.6f}\tMPJPE[STD]: {mpjpe_std:.6f}\tHRNet Loss: {hrnet_loss}')

			overall_iter += 1
			if overall_iter % config.SAVE_ITER_FREQ == 0:
				torch.save(model.state_dict(), os.path.join(config.LOG_PATH, config.NAME))

def eval_model(model, eval_loader, pretrained=False):
	if pretrained:
		model.load_state_dict(torch.load(os.path.join(config.LOG_PATH, config.NAME)))

	print("[+] Starting evaluation.")
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
	generate_submission(prediction, "submission.csv.gz")
	create_zip_code_files("code.zip")

def main():
	normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_set = DataSet(config.DATA_PATH, image_transforms=normalize, num_joints=16)
	train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS, shuffle=True)

	eval_set = DataSet(config.DATA_PATH, normalize=False, mode="valid", image_transforms=normalize)
	eval_loader = DataLoader(eval_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)

	model = models.hrnet.PoseHighResolution3D(config.hrnet, config.USE_GPU)

	print_all_attr(config)

	train_model(model, train_loader)
	eval_model(model, eval_loader, pretrained=True)

if __name__ == '__main__':
	main()