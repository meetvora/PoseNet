import torch
import config
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import os
from torch.utils.data import DataLoader
from models import *
from data import DataSet
from utils import *

if config.USE_GPU:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_model(model, train_loader):
	model.train()
	optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)
	overall_iter = 0

	for epoch in range(config.NUM_EPOCHS):
		for batch_idx, sample in enumerate(train_loader):
			image, pose3d = sample['image'], sample['pose3d']
			optimizer.zero_grad()
			output = model(image)
			loss = F.mse_loss(output, pose3d);
			loss.backward()
			optimizer.step()

			if batch_idx % 500 == 0:
				print(f'Train Epoch: {epoch} [{batch_idx}]\tLoss: {loss.item():.6f}')

			overall_iter += 1
			if overall_iter % config.SAVE_ITER_FREQ == 0:
				torch.save(model.state_dict(), os.path.join(config.LOG_PATH, config.NAME))

def eval_model(model, eval_loader, eval_set, pretrained=False):
	if pretrained:
		model.load_state_dict(os.path.join(config.LOG_PATH, config.NAME))

	with torch.no_grad():
		model.eval()
		prediction = list()

		for batch_idx, sample in enumerate(eval_loader):
			image = sample
			p3d_out = model(image)
			p3d_out = unnormalize_pose(p3d_out, eval_set.mean, eval_set.std).astype(np.int16)
			prediction = np.append(prediction, p3d_out)

	prediction = prediction.reshape(-1, 51)
	generate_submission(prediction, "submission.csv.gz")
	create_zip_code_files("code.zip")

def main():
	train_set = DataSet(config.DATA_PATH)
	train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS, shuffle=True)

	eval_set = DataSet(config.DATA_PATH, normalize=False, mode="valid")
	eval_loader = DataLoader(eval_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)

	model = config.MODEL

	train_model(model, train_loader)
	eval_model(model, eval_loader, eval_set)

if __name__ == '__main__':
	main()