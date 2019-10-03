# PoseNet

#### Distilling 3D Human Pose from a Single Image
----------------------------
Code & Report for course Machine Perception at ETH Zurich, Spring 2019.

### Reproducing Results

#### Install Requirements
```bash
make requirements
mkdir log/
```
##### Note: All configuration files provided assume that all available data is placed at `/cluster/project/infk/hilliges/lectures/mp19/project2/`

#### Training from scratch
1. Training only HRNet
 	- trains only for 2D pose estimation.
 	- applies augmentation (horizontal flip).
 	- trained for 1 epoch.
 	- makes use of `config/pretrain.py`
 	- weights stored at `log/pretrain/`
 	- if configuration unchanged, the name should be `PRETRAIN-master-Adam-1`
 	- should take around 4 hours.

```bash
make submit SCRIPT=scripts/pretrainHRN.py Q=6:00
```

2. Train entire model using pretrained-HRNet
	- uses weights from step 1 as initialization for HRNet.
	- makes use of `config/posenet.py`
	- configuration loads weights directly from step 1. In case of any issues, kindly set the variable `PRETRAINED` on line 41 of `config/pretrain.py` to the path of the step 1 weights.
	- trained for 8 epochs.
	- produces submission files after every epoch. The last one is the submitted to the leaderboard.
	- should take around additional 30 hours.

```bash
make submit SCRIPT=main.py Q=32:00
```

#### Pretrained weights
For sake of convienience, I have uploaded the pretrained weights of step 1 to [polybox](https://polybox.ethz.ch/index.php/s/YwitdaXVXN31UWB). One can download and place them at `log/pretrain/` and start directly with step 2.
```bash
wget https://polybox.ethz.ch/index.php/s/YwitdaXVXN31UWB/download -O log/pretrain/PRETRAIN-master-Adam-1
```

##### Run-time logs
Logs are flushed to file, which can be found in `log/pretrain/` for pretraining and `log/master/` for step 2.
Users can also stream the log outputs using the command `make stream DIR=<pretrain/master>`
Logs are saved in the format `{dd}-{mm}--{H}-{s}` in respective directories.
