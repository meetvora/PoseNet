import torch
import models

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
# LOG_PATH = "./log/example/"
# DATA_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"
LOG_PATH = "/home/spyd3/code/mp/skeleton/torch/log/"
DATA_PATH = "/home/spyd3/code/mp/skeleton"

SUBMISSION_FILES = [
    "data.py",
    "models",
    "utils.py",
    "config.py"
]