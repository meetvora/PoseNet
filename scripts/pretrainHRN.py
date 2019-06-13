"""Train the base HRNet using heatmaps from GT 2D coordinates"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import logging
import sys
import ipdb
import models
import models.hrnet
import config.pretrain
import config.posenet
from torch.utils.data import DataLoader
from core.data import DataSet
from core.utils import *
from core.loss import *

if config.pretrain.USE_GPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

logFormatter = "%(asctime)s - [%(levelname)s] %(message)s"
logging.basicConfig(filename=config.pretrain.LOG_NAME,
                    filemode='a',
                    format=logFormatter,
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_new_HR2D() -> nn.Module:
    """ 
	Returns an instance of HRNet with new final Conv2d layer
	"""
    model = models.hrnet.PoseHighResolutionNet(config.posenet)
    model.init_weights(config.pretrain.BASE_WEIGHTS, config.pretrain.USE_GPU)
    for param in model.parameters():
        param.requires_grad = True
    return model


def train(model: nn.Module, train_loader: DataLoader) -> None:
    model.train()
    optimizer = getattr(optim, config.pretrain.OPTIMIZER)(
        model.parameters(),
        lr=config.pretrain.LEARNING_RATE,
        weight_decay=config.pretrain.WEIGHT_DECAY)
    overall_iter = 0
    JointLoss = JointsMSELoss()

    logger.info("[+] Starting training.")
    for epoch in range(config.pretrain.NUM_EPOCHS):
        for batch_idx, sample in enumerate(train_loader):
            image, heatmap2d = sample['image'], sample['heatmap2d']
            if config.pretrain.USE_GPU:
                image, heatmap2d = to_cuda([image, heatmap2d])
            optimizer.zero_grad()
            output = model(image)
            loss = JointLoss(output, heatmap2d)
            loss.backward()
            optimizer.step()

            if batch_idx % config.pretrain.PRINT_BATCH_FREQ == 0:
                logger.debug(
                    f'Train Epoch: {epoch} [{batch_idx}]\tLoss: {loss.item():.6f}'
                )

            overall_iter += 1
            if overall_iter % config.pretrain.SAVE_ITER_FREQ == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.pretrain.LOG_PATH,
                                 config.pretrain.NAME))
    logger.info("[+] Finished training.")


def main():
    train_set = DataSet(config.DATA_PATH,
                        mode="train",
                        image_transforms=["RandomHorizontalFlip"],
                        transform_params=[(1, )])
    train_loader = DataLoader(train_set,
                              batch_size=config.pretrain.BATCH_SIZE,
                              shuffle=True)

    model = get_new_HR2D()
    print_all_attr([config, config.pretrain], logger)
    train(model, train_loader)

    torch.save(model.state_dict(),
               os.path.join(config.pretrain.LOG_PATH, config.pretrain.NAME))
    logger.info("[+] Saved final model at %s" %
                os.path.join(config.pretrain.LOG_PATH, config.pretrain.NAME))


if __name__ == '__main__':
    main()
