import torch
import models
import os
import subprocess

# Global parameters
USE_GPU = True
NUM_SAMPLES= 312188

# Hyper parameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
WORKERS = 0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
NOISE_STD = 1e-5 # Set to 0 to disable noising
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = 2000

# a_0*loss_3d + a_1*loss_2d
CYCLICAL_LOSS_COEFF = [1, 0]

# Optimizer 
# Possible amongst ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'LBFGS', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
OPTIMIZER = "RMSprop"

# Path
LOG_PATH = "./log/"
DATA_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"

SUBMISSION_FILES = [
    "data.py",
    "models",
    "utils.py",
]

if not os.path.isdir(LOG_PATH):
	os.mkdir(LOG_PATH)

BRANCH = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
NAME = "%s-%s-%s-%s" % (BRANCH, OPTIMIZER, LEARNING_RATE, BATCH_SIZE)