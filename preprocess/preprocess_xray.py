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

    xray = np.load(file_path)
    xray = np.moveaxis(np.repeat(np.expand_dims(xray, axis=0), 3, axis=0), 0, -1)

    print("Original:   ", xray.shape)

    # resize and normalize
    xray = format_xray(xray) 
    print("after formatting:   ", xray.shape)

    return xray

def preprocess_xray_directory(src_path, dst_path):

    for file_name in os.listdir(src_path):

        file_path = os.path.join(src_path, file_name)
        xray_file = file_path + '/0.npy'
        accession_number = os.path.splitext(os.path.basename(file_path))[0]
        print(accession_number)
        xray = preprocess_xray(xray_file)
        np.save(dst_path + accession_number, xray)

    return

if __name__ == "__main__":
    # Resize the scan to 128 x 192 x 192
    # Latents are reshaped to  4 x 24 x 24 x 128 
    src_path = "D:/XRayCTPA/XRay/"
    dst_path = "D:/XRayCTPA/XRay_preprocessed_224_224/"
    preprocess_xray_directory(src_path, dst_path)



