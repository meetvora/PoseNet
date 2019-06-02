import torch
import numpy as np
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

class BoneSymmMSELoss(nn.Module):
	def __init__(self, num_joints=17):
		super(BoneSymmMSELoss, self).__init__()
		H36_joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', \
			'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
		valid_pairs = [['Hip', 'Knee'], ['Knee', 'Foot'], ['Shoulder', 'Elbow'], ['Elbow', 'Wrist']]
		self.joint_map = {k: v for (v, k) in enumerate(H36_joints)}
		self.num_joints = 17
		self.valid_pairs = [(side + joint[0], side + joint[1]) for joint in valid_pairs for side in ("L", "R")]
		self.valid_pairs += [('Neck', 'LShoulder'), ('Neck', 'RShoulder'), ('Hip', 'LHip'), ('Hip', 'RHip')]

	def _get_length(self, pair, values):
		a, b = pair
		a_idx, b_idx = self.joint_map[a], self.joint_map[b]
		a_coords = values.view(-1, self.num_joints, 3)[:, a_idx]
		b_coords = values.view(-1, self.num_joints, 3)[:, b_idx]
		return torch.sum((a_coords - b_coords) ** 2, dim=1)

	def forward(self, coordinates):
		lengths = torch.stack([self._get_length(pair, coordinates) for pair in self.valid_pairs])
		lengths = torch.transpose(lengths, 0, 1)
		lengths = lengths.reshape(-1, 6, 2)
		error = torch.mean((lengths[:, :, 0] - lengths[:, :, 1]) ** 2, 1)
		return torch.mean(error)

def MPJPE_(p3d_out, p3d_gt, p3d_std):
	p3d_out_17x3 = np.reshape(p3d_out.cpu().numpy(), [-1, 17, 3])
	p3d_gt_17x3 = np.reshape(p3d_gt.cpu().numpy(), [-1, 17, 3])

	diff_std = p3d_std * (p3d_out_17x3 - p3d_gt_17x3)
	mse_std = np.square(diff_std).sum(axis=2)
	return np.mean(np.sqrt(mse_std))