import hashlib
import itertools
import math
import os
import inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from PIL import Image
from torch.utils.data import Dataset
import re

from params import *

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
import warnings
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

transform_xray = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(0.633, 0.181),])

def format_xray(img):

    # Normalize
    img = transform_xray(img).float()

    #Resize
    img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

    formated_img = img.squeeze().detach().cpu().numpy()
    return formated_img

def preprocess_xray(file_path):

    ds = pydicom.dcmread(file_path)
    xray = ds.pixel_array.astype(np.float32)  # Convert to float32
    xray /= np.max(xray)  # Normalize to [0, 1] range

    xray = np.moveaxis(np.repeat(np.expand_dims(xray, axis=0), 3, axis=0), 0, -1)

    print("Original:   ", xray.shape)

    # resize and normalize
    xray = format_xray(xray) 
    print("after formatting:   ", xray.shape)

    return xray

def preprocess_xray_directory(xray_csv_path, src_path, dst_path):

    df_train = pd.read_csv(xray_csv_path + 'train.csv')
    df_test = pd.read_csv(xray_csv_path + 'test.csv')
    df = pd.concat([df_train, df_test], ignore_index=True)

    for index, row in df.iterrows():
        if row['Modality'] in ('CR', 'DX'):

            accession_number = row['Subject ID']  # Here you can use the exact column name as it appears in df.columns
            folder_path = dst_path + accession_number

            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            dcm_location = 'E:\\LIDC\\' + row['File Location'] + '\\'
            idx = 0
            for filename in os.listdir(dcm_location):
                if filename.endswith('.dcm'):

                    file_path = os.path.join(dcm_location, filename)
                    print(f"Processing file: {file_path}")
                    print(idx)
                    xray_file = folder_path + '/' + str(idx) +'.npy'
                    xray = preprocess_xray(file_path)
                    np.save(xray_file, xray)
                    idx += 1

    return

if __name__ == "__main__":
    # Resize the scan to 128 x 192 x 192
    # Latents are reshaped to  4 x 24 x 24 x 128 
    xray_csv_path = "E:/LIDC/cross_validation_ct_xray/fold0/"
    src_path = "E:/LIDC/LIDC-IDRI/"
    dst_path = "D:/LIDC/XRay_preprocessed_224_224/"
    preprocess_xray_directory(xray_csv_path, src_path, dst_path)



