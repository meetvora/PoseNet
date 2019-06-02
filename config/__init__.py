import torch
import models
import os
import subprocess
import datetime
import logging
import sys

# Hyper parameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
__NOISE_STD = 1e-5 # Set to 0 to disable noising
PRINT_BATCH_FREQ = 10
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = 2000
# Possible amongst ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'LBFGS', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
OPTIMIZER = "Adam"

# Experiment parameters
USE_GPU = True
NUM_JOINTS = 17
BRANCH = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
NAME = "%s-%s-%s_%s" % (BRANCH, OPTIMIZER, BATCH_SIZE, NUM_JOINTS)

LOG_PATH = "./log/%s" % BRANCH
DATA_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"

__SUBMISSION_FILES__ = [
    "data.py",
    "models",
    "utils.py",
    "main.py",
    "README.md",
    "requirements.txt",
    "scripts"
]

if not os.path.isdir(LOG_PATH):
	os.mkdir(LOG_PATH)

LOG_NAME = os.path.join(LOG_PATH, f"{datetime.datetime.now().strftime('%d-%m--%H-%M')}.log")
__LOG_TO_FILE__ = True
__logFormatter__ = "%(asctime)s - [%(levelname)s] %(message)s"
if __LOG_TO_FILE__:
	__LOG_PARAMS__ = {
		'filename':LOG_NAME,
		'filemode':'a',
		'format':logFormatter,
		'level':logging.DEBUG}
else:
	__LOG_PARAMS__ = {
		'format':logFormatter,
		'level':logging.DEBUG,
		'stream':sys.stdout
	}