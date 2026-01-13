from torchvision import transforms
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from flautim.pytorch.Dataset import Dataset
import flautim as fl
from torch.utils.data import random_split, DataLoader
import numpy as np
import copy
import pandas as pd
from torchvision.transforms import v2 as tranformsv2
from torchvision.transforms.v2 import functional as func

class SypakDataset(Dataset):
    
    def __init__(self, df, train_transforms = None, eval_transforms = None):
        super(SypakDataset, self).__init__(name = "Sypak")
        self.df = df
        self.X = np.stack(df.drop('label', axis=1).values).astype(np.uint8)
        self.y = df['label'].astype('category').cat.codes.values
        self.train = True
        self.train_transforms = train_transforms if train_transforms else transforms.Lambda(lambda x: x)
        self.eval_transforms = eval_transforms if eval_transforms else transforms.Lambda(lambda x: x)
        
        torch.manual_seed(42)
        total_size = len(self.df)
        eval_size = int(total_size * 0.2)
        train_size = total_size - eval_size
        
        self.train_subset, self.eval_subset = random_split(
            range(total_size), [train_size, eval_size]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).reshape(3, 224, 224) / 255.0 
        y = torch.tensor(self.y[idx], dtype=torch.long) 
        x = func.to_image(x)
        transforms = self.train_transforms if self.train else self.eval_transforms
        x = transforms(x)
        return x, y
    
    def dataloader(self, validation=False, batch_size=32, num_workers=0):
        self.train = not validation
        indices = self.eval_subset if validation else self.train_subset
        subset = torch.utils.data.Subset(self, indices)
        return DataLoader(subset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)