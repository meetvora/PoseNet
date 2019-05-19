import os
import subprocess
import config

USE_GPU = config.USE_GPU

GAUS_KERNEL = 3
GAUS_STD = 2

OPTIMIZER = "Adam"
MOMENTUM = 0.9
NESTEROV = False

# Path
LOG_PATH = "./log/finetune/"
DATA_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"

NUM_EPOCHS = 1
BATCH_SIZE = 32
WORKERS = 0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
NOISE_STD = 1e-5 # Set to 0 to disable noising
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = 2000

BRANCH = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
NAME = "%s-%s-%s-%s" % ("FINETUNE", BRANCH, OPTIMIZER, LEARNING_RATE)

if not os.path.isdir(LOG_PATH):
	os.mkdir(LOG_PATH)