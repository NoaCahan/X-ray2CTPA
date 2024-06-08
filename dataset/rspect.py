import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import glob
import pandas as pd

RSPECT_CT_STUDY_COL = "StudyInstanceUID"
RSPECT_CT_SERIES_COL = "SeriesInstanceUID"
RSPECT_LABEL_COL = "negative_exam_for_pe"

class RSPECTDataset(Dataset):
    def __init__(self, root_dir='../RSPECT/', target=None, mode="train", augmentation=False):
        if target is None and mode != "infer":
            raise(RuntimeError("both images and targets must be set if mode is not 'infer'"))

        if target is not None:
            self.data = pd.read_csv(target)

        self.mode = mode
        self.root_dir = root_dir
        self.VAE = True
        self.text_label = False
        self.cts = self.root_dir
        self.xray_stub = '../datasets/ctpa_xray/XRay_preprocessed/4015005309959.npy'
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ct_study = self.data.loc[idx, RSPECT_CT_STUDY_COL]
        ct_series = self.data.loc[idx, RSPECT_CT_SERIES_COL]
        ct =  np.load(self.cts + str(ct_study) + '_' + str(ct_series) + '.npy').astype(np.float32)

        if self.mode == "train":

            label = 1 - self.data.loc[idx, RSPECT_LABEL_COL]
            if not self.text_label:
                label = torch.tensor(label).type(torch.DoubleTensor)
                label = label.reshape(1)
            else:
                if label:
                    label = "Positive"
                else: 
                    label = "Negative"
        else: 
            label = 2
            if not self.text_label:
                label = torch.tensor(label).type(torch.DoubleTensor)
                label = label.reshape(1)
            else:
                label = "Unknown"

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

        # A constant xray scan to be used as a stub 
        # Load matching Xray 2D image
        xray = torch.from_numpy(np.load(self.xray_stub)).float()

        if self.mode == "train" or self.mode == "test":
            return {'ct': ctout, 'cxr': xray, 'target': label} 
        else: #if self.mode == "infer"
            return {'ct': ctout, 'cxr': xray, 'ct_study': ct_study, 'ct_series': ct_series, 'cxr_accession': cxr_accession}
