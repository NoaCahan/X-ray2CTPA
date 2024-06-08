import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import glob


class LIDCDataset(Dataset):
    def __init__(self, root_dir='../LIDC', augmentation=False):
        self.root_dir = root_dir 
        self.file_names = glob.glob(os.path.join(
            root_dir, './**/*.npy'), recursive=True)
        self.augmentation = augmentation
        self.xray_stub = '../datasets/ctpa_xray/XRay_preprocessed/4015005309959.npy'

        self.VAE = True

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        img = np.load(path).astype(np.float32)

        if not self.VAE:
            if self.augmentation:
                random_n = torch.rand(1)
                if random_n[0] > 0.5:
                    img = np.flip(img, 2)

        imageout = torch.from_numpy(img.copy()).float()
        # For VAE
        if self.VAE:
            imageout = imageout.permute(0,3,1,2)
        else:
            imageout = imageout.unsqueeze(0).permute(0,3,1,2)

        xrayout = torch.from_numpy(np.load(self.xray_stub)).float()

        # Adding a stub label as well as a stub xray
        label = torch.tensor(2).type(torch.DoubleTensor).reshape(1)

        return {'name': 'LIDC', 'ct': imageout, 'cxr': xrayout, 'target': label}