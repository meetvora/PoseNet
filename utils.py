import torch
import numpy as np
import patoolib
import config


def unnormalize_pose(p3d, p3d_mean, p3d_std):
    b = p3d.shape[0]
    if config.USE_GPU:
        p3d, p3d_mean, p3d_std = to_cuda([p3d, p3d_mean, p3d_std])

    p3d_17x3 = torch.reshape(p3d, [-1, 17, 3])
    root_joint = p3d_17x3[:, 0, :]
    root_joint = torch.unsqueeze(root_joint, 1)
    root_joint = root_joint.repeat([1, 17, 1])
    p3d_17x3 = p3d_17x3 - root_joint
    p3d_17x3 = p3d_17x3 * p3d_std[:b, ...] + p3d_mean[:b, ...]
    p3d = torch.reshape(p3d_17x3, [-1, 51])
    return p3d.cpu().numpy()


def generate_submission(predictions, out_path):
    ids = np.arange(1, predictions.shape[0] + 1).reshape([-1, 1])

    predictions = np.hstack([ids, predictions])

    joints = [
        'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine',
        'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist'
    ]
    header = ["Id"]

    for j in joints:
        header.append(j + "_x")
        header.append(j + "_y")
        header.append(j + "_z")

    header = ",".join(header)
    np.savetxt(out_path,
               predictions,
               delimiter=',',
               header=header,
               comments='')


def create_zip_code_files(output_file):
    patoolib.create_archive(output_file, config.SUBMISSION_FILES)


def print_all_attr(modules, logger):
    sep = "-" * 90
    logger.info(sep)
    for module in modules:
        attr = [i for i in dir(module) if ("__" not in i and i.upper() == i)]
        pairs = {k: getattr(module, k) for k in attr}
        logger.info(f"{module.__name__}")
        logger.info(f"{sep}")
        for at in attr:
            logger.info(f"{at} ==> {pairs[at]}")
        logger.info(sep)


to_cuda = lambda u: map(lambda x: x.cuda(), u)


def print_termwise_loss(loss):
    keys = loss.keys()
    string = "=> Individual loss >>\t" + ("\t".join(
        [f"{k.upper()}: {v:.6f}" for (k, v) in loss.items()]))
    return string + "\n"
