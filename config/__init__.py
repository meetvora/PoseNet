import torch
import models
import os
import subprocess
import datetime
import logging
import sys

# Hyper parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
PRINT_BATCH_FREQ = 10
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = None
# Possible amongst ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'LBFGS', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
OPTIMIZER = "Adam"

# Experiment parameters
__PRODUCTION__ = True

USE_GPU = __PRODUCTION__
NUM_JOINTS = 17
BRANCH = subprocess.check_output(["git", "rev-parse", "--abbrev-ref",
                                  "HEAD"]).strip().decode("utf-8")
NAME = "%s-%s-%s_%s" % (BRANCH, OPTIMIZER, BATCH_SIZE, NUM_JOINTS)

LOG_PATH = "./log/%s" % BRANCH
DATA_PATH = "/cluster/project/infk/hilliges/lectures/mp19/project2/"

__SUBMISSION_FILES__ = [
    "data.py", "models", "utils.py", "main.py", "README.md",
    "requirements.txt", "scripts"
]

if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)

# Logging configuration
LOG_NAME = os.path.join(
    LOG_PATH, f"{datetime.datetime.now().strftime('%d-%m--%H-%M')}.log")

__logFormatter__ = "%(asctime)s - [%(levelname)s] %(message)s"

if __PRODUCTION__:
    __LOG_PARAMS__ = {
        'filename': LOG_NAME,
        'filemode': 'a',
    }
else:
    __LOG_PARAMS__ = {
        'stream': sys.stdout,
    }

__LOG_PARAMS__.update({
    'format': __logFormatter__,
    'level': logging.DEBUG,
    'datefmt': '%d/%m/%Y %H:%M:%S'
})
