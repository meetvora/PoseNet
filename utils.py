import torch
import numpy as np
import patoolib
import config
import torch.nn as nn

class JointsMSELoss(nn.Module):
	"""A custom loss function developed to learn HRNet.
	Calculates MSE between GT and Predict heatmaps"""
	def __init__(self, use_target_weight=False):
		super(JointsMSELoss, self).__init__()
		self.criterion = nn.MSELoss(reduction='mean')
		self.use_target_weight = use_target_weight

	def forward(self, output, target, target_weight=None):
		batch_size = output.size(0)
		num_joints = output.size(1)
		heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
		heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
		loss = 0

		for idx in range(num_joints):
			heatmap_pred = heatmaps_pred[idx].squeeze()
			heatmap_gt = heatmaps_gt[idx].squeeze()
			if self.use_target_weight:
				loss += 0.5 * self.criterion(
					heatmap_pred.mul(target_weight[:, idx]),
					heatmap_gt.mul(target_weight[:, idx]))
			else:
				loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

		return loss / num_joints

def compute_MPJPE(p3d_out, p3d_gt, p3d_std):
	p3d_out_17x3 = np.reshape(p3d_out.cpu().numpy(), [-1, 17, 3])
	p3d_gt_17x3 = np.reshape(p3d_gt.cpu().numpy(), [-1, 17, 3])

	diff_std = p3d_std * (p3d_out_17x3 - p3d_gt_17x3)
	mse_std = np.square(diff_std).sum(axis=2)
	return np.mean(np.sqrt(mse_std))

def unnormalize_pose(p3d, p3d_mean, p3d_std):
	b = p3d.shape[0]
	if config.USE_GPU:
		p3d, p3d_mean, p3d_std = to_cuda([p3d, p3d_mean, p3d_std])

	p3d_17x3 = torch.reshape(p3d, [-1, 17, 3])
	root_joint = p3d_17x3[:, 0, :]
	root_joint = torch.unsqueeze(root_joint, 1)
	root_joint = root_joint.repeat([1, 17, 1])
	p3d_17x3 = p3d_17x3 - root_joint
	p3d_17x3 = p3d_17x3 * p3d_std[:b,...] + p3d_mean[:b,...]
	p3d = torch.reshape(p3d_17x3, [-1,51])
	return p3d.cpu().numpy()

def generate_submission(predictions, out_path):
	ids = np.arange(1, predictions.shape[0] + 1).reshape([-1, 1])

	predictions = np.hstack([ids, predictions])

	joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head',
	          'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
	header = ["Id"]

	for j in joints:
		header.append(j + "_x")
		header.append(j + "_y")
		header.append(j + "_z")

	header = ",".join(header)
	np.savetxt(out_path, predictions, delimiter=',', header=header, comments='')

def create_zip_code_files(output_file):
	patoolib.create_archive(output_file, config.SUBMISSION_FILES)

def print_all_attr(modules, logger):
	sep = "-"*90
	logger.info(sep)
	for module in modules:
		attr = [i for i in dir(module) if ("__" not in i and i.upper() == i)]
		pairs = {k: getattr(module, k) for k in attr}
		print(f"{module.__name__}\n{sep}")
		for at in attr:
			logger.info(f"{at} ==> {pairs[at]}")
		logger.info(sep)

to_cuda = lambda u: map(lambda x: x.cuda(), u)

def print_termwise_loss(loss):
	keys = loss.keys()
	string = "=> Individual loss >>\t" + ("\t".join([f"{k.upper()}: {v:.6f}" for (k, v) in loss.items()]))
	return string+"\n"