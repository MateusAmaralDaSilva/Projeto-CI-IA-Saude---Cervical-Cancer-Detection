import torch
import torch.nn as nn
from torchvision import models
from flautim.pytorch.Model import Model

class ResNet50Classifier(Model):
    def __init__(self, context, num_classes, **kwargs):
        super().__init__(context, name="ResNet50", version=1, id=1, **kwargs)

        model = models.resnet50(pretrained=True)

        self.embedding_dimension = model.fc.in_features

        model.fc = nn.Identity()
        self.backbone = model

        self.classifier = nn.Linear(self.embedding_dimension, num_classes)

    def forward(self, x, return_embeddings=False):
        emb = self.backbone(x)
        logits = self.classifier(emb)

        if return_embeddings:
            return logits, emb

        return logits

    def get_embeddings(self, x):
        return self.backbone(x)
