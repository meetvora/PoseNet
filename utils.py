import torch
import numpy as np
import patoolib
import config

def compute_MPJPE(p3d_out, p3d_gt, p3d_std):
	p3d_out_17x3 = np.reshape(p3d_out.cpu().numpy(), [-1, 17, 3])
	p3d_gt_17x3 = np.reshape(p3d_gt.cpu().numpy(), [-1, 17, 3])

	diff = (p3d_out_17x3 - p3d_gt_17x3)
	diff_std = diff * p3d_std
	mse, mse_std = np.square(diff).sum(axis=2), np.square(diff_std).sum(axis=2)
	return np.mean(np.sqrt(mse)), np.mean(np.sqrt(mse_std))

def unnormalize_pose(p3d, p3d_mean, p3d_std):
	b = p3d.shape[0]
	if config.USE_GPU:
		p3d = p3d.cuda()
		p3d_mean = p3d_mean.cuda()
		p3d_std = p3d_std.cuda()

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

def print_all_attr(module):
	attr = [i for i in dir(module) if "__" not in i]
	pairs = {k: getattr(module, k) for k in attr}
	sep = "-"*90
	print(f"{module.__name__}\n{sep}")
	for at in attr:
		print(f"{at} ==> {pairs[at]}")
	print(sep)
