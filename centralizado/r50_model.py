import torch
import torch.nn as nn
from torchvision import models
from flautim.pytorch.Model import Model

class ResNet50Classifier(Model):
    def __init__(self, context, num_classes, **kwargs):
        super(ResNet50Classifier, self).__init__(context, name = "ResNet50", version = 1, id = 1, **kwargs)
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
