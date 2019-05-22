import os
import subprocess
import config

USE_GPU = config.USE_GPU
NUM_JOINTS = config.NUM_JOINTS
#(True, False) for [END_TO_END, SOFTARGMAX] is not possible
END_TO_END, SOFTARGMAX = (True, True)

# Term-wise loss coefficients
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
	'IN_SIZE': NUM_JOINTS * 2,
	'OUT_SIZE': 17 * 3
}
THREED = {
	'LINEAR_SIZE': 1024,
	'NUM_BLOCKS': 2,
	'p': 0.5,
	'IN_SIZE': 17 * 3,
	'OUT_SIZE': NUM_JOINTS * 2
}

# HRNet Parameters
# Points to weights stored by scripts/finetuneHRN.py or pre-trained MPII weights (if NUM_JOINTS = 16)
if NUM_JOINTS == 16:
	PRETRAINED = '/cluster/home/voram/mp/PoseNet/models/weights/pose_hrnet_w32_256x256.pth'
else:
	PRETRAINED = os.path.join(config.finetune.LOG_PATH, config.finetune.NAME)
INIT_WEIGHTS = True
TARGET_TYPE = config.hrnet.TARGET_TYPE
IMAGE_SIZE = config.hrnet.IMAGE_SIZE
HEATMAP_SIZE = config.hrnet.HEATMAP_SIZE
SIGMA = config.hrnet.SIGMA
EXTRA = config.hrnet.EXTRA

# Parameters for generating heatmaps from coordinates
GAUS_KERNEL = 3
GAUS_STD = 1