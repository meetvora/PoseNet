import torch
import models
import os

# Global parameters
USE_GPU = True
NUM_SAMPLES= 312188
DEMO = True
NAME = "cyclical"
# MODEL = models.demo_model

# Hyper parameters
NUM_EPOCHS = 10
BATCH_SIZE = 16
WORKERS = 4
DENOISE = True
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
NOISE_STD = 1e-5
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = 2000

# a_0*loss_3d + a_1*loss_2d
CYCLICAL_LOSS_COEFF = [1, 0]

# Path
LOG_PATH = "./log/example/"
DATA_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"

SUBMISSION_FILES = [
    "data.py",
    "models",
    "utils.py",
    "config.py"
]

if not os.path.isdir(LOG_PATH):
	os.mkdir(LOG_PATH)
