import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
regressor_to17x3 = torch.nn.Linear(1000, 51)
demo_model = torch.nn.Sequential(resnet18, 
				torch.nn.ReLU(),
				regressor_to17x3)