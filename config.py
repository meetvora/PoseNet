import torch
import models
import os

# Global parameters
USE_GPU = True
NUM_SAMPLES= 312188
DEMO = True
NAME = "demo"
MODEL = demo_model

# Hyper parameters
NUM_EPOCHS = 5
BATCH_SIZE = 1
WORKERS = 1
LEARNING_RATE = 0.003
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = 2000

# Path
LOG_PATH = "log/"
DATA_PATH = None

SUBMISSION_FILES = [
    "data.py",
    "models",
    "utils.py",
    "config.py"
]

if not os.path.isdir(LOG_PATH):
	os.mkdir(LOG_PATH)