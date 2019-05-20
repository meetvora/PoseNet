import os
import subprocess
import config
import config.finetune
import config.hrnet

USE_GPU = config.USE_GPU
OPTIMIZER = "Adam"
FROZEN = True

# Path
LOG_PATH = "./log/posenet/"
DATA_PATH = config.DATA_PATH

NUM_EPOCHS = 5
BATCH_SIZE = 32
WORKERS = 0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = 2000

BRANCH = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
NAME = "%s-%s-%s-%s" % (BRANCH, OPTIMIZER, LEARNING_RATE, BATCH_SIZE)

if not os.path.isdir(LOG_PATH):
	os.mkdir(LOG_PATH)

LOSS_COEFF = {
	'hrnet_maps': 0,
	'cycl_martinez':{
		'pose_3d': 1,
		'pose_2d': 0
	}
}

# Martinez Parameters
TWOD = {
	'LINEAR_SIZE': 1024,
	'NUM_BLOCKS': 2,
	'p': 0.5,
	'IN_SIZE': 17 * 2,
	'OUT_SIZE': 17 * 3
}
THREED = {
	'LINEAR_SIZE': 1024,
	'NUM_BLOCKS': 2,
	'p': 0.5,
	'IN_SIZE': 17 * 3,
	'OUT_SIZE': 17 * 2
}

# HRNet Parameters
# Points to weights stored by scripts/finetuneHRN.py
PRETRAINED = '/cluster/home/voram/mp/PoseNet/log/finetune/FINETUNE-softargmax-Adam-5'
#PRETRAINED = os.path.join(config.finetune.LOG_PATH, config.finetune.NAME)
NUM_JOINTS = 17
INIT_WEIGHTS = True
TARGET_TYPE = config.hrnet.TARGET_TYPE
IMAGE_SIZE = config.hrnet.IMAGE_SIZE
HEATMAP_SIZE = config.hrnet.HEATMAP_SIZE
SIGMA = config.hrnet.SIGMA
EXTRA = config.hrnet.EXTRA

# Parameters for generating heatmaps from coordinates
GAUS_KERNEL = 3
GAUS_STD = 2
