import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import glob
import pandas as pd
from params import *

class XrayLIDCDataset(Dataset):
    def __init__(self, root_dir='.', target=None, mode="train", augmentation=False):
        if target is None and mode != "infer":
            raise(RuntimeError("both images and targets must be set if mode is not 'infer'"))

        if target is not None:
            df = pd.read_csv(root_dir + target)
            self.data = df[df['Modality'].isin(['CT'])]
            self.data.reset_index(drop=True, inplace=True)
        self.mode = mode
        self.root = root_dir
        self.cts = self.root + 'lidc_vae_256/'

        self.xrays = self.root + 'XRay_preprocessed_224_224/'
        self.augmentation = augmentation
        self.VAE = True
        self.text_label = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        accession = self.data.loc[idx, LIDC_CT_ACCESSION_COL]
        label = 2

        # Load the CTPA 3D scan
        ct =  np.load(self.cts + accession + '.npy').astype(np.float32)

        if not self.VAE:
            if self.augmentation:
                random_n = torch.rand(1)
                if random_n[0] > 0.5:
                    ct = np.flip(ct, 0)

        ctout = torch.from_numpy(ct.copy()).float()
        if not self.VAE:
            ctout = ctout.unsqueeze(0)
        else:
            ctout = ctout.permute(0,3,1,2)

        # Load matching Xray 2D image
        xray = torch.from_numpy(np.load(self.xrays + accession + '.npy')).float()

        if self.mode == "train" or self.mode == "test":
            return {'ct': ctout, 'cxr': xray, 'target': label} 
        else: #if self.mode == "infer"
            return {'ct': ctout, 'cxr': xray, 'accession': accession}