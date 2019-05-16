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
import config
from data import DataSet, LiftingDataSet
from utils import *
from models.lifting import CyclicalMartinez

if config.USE_GPU:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_model(model, train_loader):
	model.train()
	optimizer = getattr(optim, config.OPTIMIZER)(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
	overall_iter = 0

	print("[+] Starting training.")
	for epoch in range(config.NUM_EPOCHS):
		for batch_idx, sample in enumerate(train_loader):
			pose2d, pose3d = sample['pose2d'].cuda(), sample['pose3d'].cuda()
			optimizer.zero_grad()
			noise = torch.from_numpy(np.random.normal(scale=config.NOISE_STD, size=pose2d.shape).astype(np.float32))
			inp = pose2d + noise.cuda()
			output3d, output2d = model(inp)
			loss = config.CYCLICAL_LOSS_COEFF[0] * F.mse_loss(output3d, pose3d) + config.CYCLICAL_LOSS_COEFF[1] * F.mse_loss(output2d, pose2d)
			loss.backward()
			optimizer.step()

			if batch_idx % 500 == 0:
				mpjpe, mpjpe_std = compute_MPJPE(output3d.detach(), pose3d.detach(), train_loader.dataset.std.numpy())
				print(f'Train Epoch: {epoch} [{batch_idx}]\tLoss: {loss.item():.6f}\tMPJPE: {mpjpe:.6f}\tMPJPE[STD]: {mpjpe_std:.6f}')

			overall_iter += 1
			if overall_iter % config.SAVE_ITER_FREQ == 0:
				torch.save(model.state_dict(), os.path.join(config.LOG_PATH, config.NAME))

def eval_model(model, eval_loader, pretrained=False):
	if pretrained:
		model.load_state_dict(os.path.join(config.LOG_PATH, config.NAME))

	print("[+] Starting evaluation.")
	with torch.no_grad():
		model.eval()
		prediction = list()

		for batch_idx, sample in enumerate(eval_loader):
			image = sample
			p3d_out = model(image)
			p3d_out = unnormalize_pose(p3d_out, eval_loader.dataset.mean, eval_loader.dataset.std).astype(np.int16)
			prediction = np.append(prediction, p3d_out)

	prediction = prediction.reshape(-1, 51)
	generate_submission(prediction, "submission.csv.gz")
	create_zip_code_files("code.zip")

def main():
	train_set = LiftingDataSet(config.DATA_PATH)
	train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS, shuffle=True)

	eval_set = LiftingDataSet(config.DATA_PATH, normalize=False, mode="valid")
	eval_loader = DataLoader(eval_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)
	model = models.lifting.CyclicalMartinez()
	model.apply(models.lifting.weight_init)

	print_all_attr(config)

	train_model(model, train_loader)
	eval_model(model, eval_loader, eval_set)

if __name__ == '__main__':
	main()