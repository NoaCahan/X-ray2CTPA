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
#import pandas as pd
#import random
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
import warnings
import scipy
from scipy.ndimage import zoom

#import vtk
#from vtk.util import numpy_support
import cv2
import glob
from scipy import ndimage
import pydicom as dicom
from lungmask import mask
from diffusers import AutoencoderKL

# Hounsfield Units for Air
AIR_HU_VAL = -1000.

# Statistics for Hounsfield Units
CONTRAST_HU_MIN = -200.     # Min value for loading contrast
CONTRAST_HU_MAX = 500.      # Max value for loading contrast
CONTRAST_HU_MEAN = 0.15897  # Mean voxel value after normalization and clipping
CONTRAST_HU_STD = 0.19974   # Standard deviation of voxel values after normalization and clipping
ct_xray_manual_accession = ['4015010182925']

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def get_series_info(dicom_file):
    ds = dicom.dcmread(dicom_file)
    series_info = {
        'SeriesInstanceUID': ds.SeriesInstanceUID,
        'SeriesDescription': ds.SeriesDescription
    }

    try:
        series_info['SeriesOrientation'] = ds.ImageOrientationPatient
    except:
        series_info['SeriesOrientation'] = 'NA'
    return series_info

def get_first_of_dicom_field_as_int(x):
    if type(x) == dicom.multival.MultiValue: return int(x[0])
    else: return int(x)

def make_rgb(volume):
    """Tile a NumPy array to make sure it has 3 channels."""
    z, c, h, w = volume.shape

    tiling_shape = [1]*(len(volume.shape))
    tiling_shape[1] = 3
    np_vol = torch.tile(volume, tiling_shape)
    return np_vol

def encode_ctpa(ct, vae):

    device = "cuda"
    weight_dtype =  torch.float32
    ct = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0)

    print("ct before vae = ", ct.shape)
    vae.eval()

    latents = []
    with torch.no_grad():
        for i in range(ct.shape[2]):
            slice = make_rgb(ct[:,:,i,:,:])
            latent = vae.encode(slice.to(device, dtype=weight_dtype)).latent_dist.sample()
            latents.append(latent)
        latents = torch.stack(latents, dim=4).squeeze()#.unsqueeze(0)#.permute(0,2,3,4,1)

        latents = latents.detach().cpu().numpy()
        print("latents shape", latents.shape)
        latents = latents * 0.18215

    return latents

def normalize(img):
    
    img = img.astype(np.float32)
    img = (img - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN) 
    img = np.clip(img, 0., 1.) * 2 -1
    return img

def resample_volume(volume, current_spacing, new_spacing):
    resize_factor = np.array(current_spacing) / new_spacing
    new_real_shape = volume.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / volume.shape
    new_spacing = np.array(current_spacing) / real_resize_factor

    print("new sampling = ", new_spacing)
    resampled_volume = scipy.ndimage.interpolation.zoom(volume, real_resize_factor)
    return resampled_volume

def resize_volume(tensor, output_size):

    z, h, w = tensor.shape
    resized_scan = np.zeros((output_size[0], output_size[1], output_size[2]))

    volume = tensor[:,:,:].squeeze()

    real_resize_factor = np.array(output_size) / np.shape(volume)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized_scan[:,:,:] = scipy.ndimage.interpolation.zoom(volume, real_resize_factor, mode='nearest').astype(np.int16)

    return resized_scan

def find_largest_connected_components(image):
    # Find connected components in the binary image
    _, labels, stats, _ = cv2.connectedComponentsWithStats(image)

    # Sort the connected components by area in descending order
    sorted_stats = sorted(stats, key=lambda x: x[4], reverse=True)

    # Get the labels of the two largest connected components (excluding background)
    largest_labels = [sorted_stats[i][0] for i in range(1, min(3, len(sorted_stats)))]

    # Create a mask with the two largest connected components
    mask = np.isin(labels, largest_labels).astype(np.uint8)

    return mask

def get_lungs_voi(scan):
    '''
    This funtion finds the lungs VOI from the given 3D scan.
    '''
    # Segment lungs and create lung mask with one label
    lungs = mask.apply(scan)
    lung_pixels = np.nonzero(lungs)
    if len(lung_pixels[0]) == 0 :
        return  np.full(scan.shape, FILLIN_VALUE_SCAN)
        

    # Get values from lung segmentation
    min_z = np.amin(lung_pixels[0])
    max_z = np.amax(lung_pixels[0])
    min_x = np.amin(lung_pixels[1])
    max_x = np.amax(lung_pixels[1])
    min_y = np.amin(lung_pixels[2])
    max_y = np.amax(lung_pixels[2])

    # Crop scan to lung VOI
    cropped_scan = scan[min_z:max_z, min_x:max_x, min_y:max_y]
    cropped_lungs = lungs[min_z:max_z, min_x:max_x, min_y:max_y]
    return cropped_scan, cropped_lungs

def preprocess_ctpa(img, attr):
    """reshape, normalize image and convert to tensor"""
    img_cropped, lungs = get_lungs_voi(img)
    print("Cropping scan to lung VOI:   ", img_cropped.shape)
    z, x,y = img_cropped.shape
    if (z and x and y) == 0:
        print(" failed to compute voi : compute manually")
        return 0,0

    # Resample
    spacing = attr['Spacing']
    img_resampled = resample_volume(img_cropped, spacing, [1,1,1])

    # Rescale

    img_resize = resize_volume(img_resampled, [128, 256, 256])

    # noramlize and window Hounsfield Units 
    img_normalized = normalize(img_resize)
    return img_normalized

def calculate_current_spacing(slices):
    spacings = []

    for slice in slices:
        pixel_spacing = slice.PixelSpacing
        slice_thickness = slice.SliceThickness

        # Convert the spacing values to floats
        pixel_spacing = [float(spacing) for spacing in pixel_spacing]
        slice_thickness = float(slice_thickness)

        spacings.append((pixel_spacing[0], pixel_spacing[1], slice_thickness))

    spacing_CHECK = [float(slices[0].SliceThickness), 
                        float(slices[0].PixelSpacing[0]), 
                        float(slices[0].PixelSpacing[0])]

    return spacing_CHECK

# Either path or slices should be given
def load_scan(path=None, slices=None):

    # Two Sorting options: 'InstanceNumber', 'SliceLocation'
    attr = {}
    if slices == None:
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    else:
        slices = [dicom.dcmread(s) for s in slices]
    # For missing ImagePositionPatient
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    slices2 = []
    prev = -1000000
    # remove redundant slices
    for slice in slices:
        # For missing ImagePositionPatient
        cur = slice.ImagePositionPatient[2]

        if cur == prev:
            continue
        prev = cur
        slices2.append(slice)
    slices = slices2

    for i in range(len(slices)-1):
        try:
            slice_thickness = np.abs(slices[i].ImagePositionPatient[2] - slices[i+1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[i].SliceLocation - slices[i+1].SliceLocation)
        if slice_thickness != 0:
            break

    spacing = calculate_current_spacing(slices)   

    for s in slices:
        s.SliceThickness = slice_thickness

    x, y = slices[0].PixelSpacing

    if slice_thickness == 0:
        attr['Spacing'] = spacing[0]
    else:
        attr['Spacing'] = (slice_thickness, x, y)

    attr['Position'] = slices[0].ImagePositionPatient
    attr['Orientation'] = slices[0].ImageOrientationPatient

    return (slices, attr)

def dicom_load_scan(paths):
    attr = {}
    slices = [dicom.read_file(path) for path in paths]
    slices.sort(key = lambda x: int(x.InstanceNumber), reverse = True)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness

    spacing = calculate_current_spacing(slices)   

    for s in slices:
        s.SliceThickness = slice_thickness

    x, y = slices[0].PixelSpacing

    if slice_thickness == 0:
        attr['Spacing'] = spacing[0]
    else:
        attr['Spacing'] = (slice_thickness, x, y)

    attr['Position'] = slices[0].ImagePositionPatient
    attr['Orientation'] = slices[0].ImageOrientationPatient

    window_center, window_width, intercept, slope = get_windowing(slices[0])
    attr['window_center'] = window_center
    attr['window_width'] = window_width
    attr['intercept'] = intercept
    attr['slope'] = slope
    
    return (slices, attr)

def get_pixels_hu(slices):

    #print([s.pixel_array.shape for s in slices])
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0#-1024

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def dicom_get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def get_high_resolusion_dicom_series(folder_path):
    series_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            dicom_file = os.path.join(root, file)
            if not dicom.dcmread(dicom_file).get('SeriesInstanceUID'):
                continue
            
            series_info = get_series_info(dicom_file)
            series_uid = series_info['SeriesInstanceUID']

            if series_uid not in series_dict:
                series_dict[series_uid] = []

            if series_info['SeriesOrientation'] != 'NA':
                if np.allclose(np.array(series_info['SeriesOrientation']).astype(float), [1., 0., 0., 0., 1., 0.], rtol=1e-3):
                   series_dict[series_uid].append(dicom_file)

    longest_series = max(series_dict.values(), key=len)
    print(series_info['SeriesOrientation'])
    print(len(longest_series))
    return longest_series

def preprocess_ctpa_directory(src_path, dst_path):

    device = DEVICE
    weight_dtype = torch.float32

    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    vae = AutoencoderKL.from_single_file(url)
    vae.to(device, dtype=weight_dtype)

    for accession_number in os.listdir(src_path):

        scan = src_path + accession_number
        scan_dst = dst_path + accession_number

        if os.path.isfile(scan_dst + '.npy'):
            print("File exists - ", scan_dst + '.npy')
        else:
            print("processing scan - ", scan)
            if accession_number in ct_xray_manual_accession:
                continue
            highres_series = get_high_resolusion_dicom_series(scan)

            slices, attr = dicom_load_scan(highres_series)
            ct = dicom_get_pixels_hu(slices)
            print("after loading exam = ", ct.shape)
            ct = preprocess_ctpa(ct, attr)
            print("after preprocessing = ", ct.shape)

            latents = encode_ctpa(ct, vae)
            print("scan_dst = ", scan_dst)

            #sitk.WriteImage(sitk.GetImageFromArray(latents), scan_dst + '.nii')
            np.save(scan_dst , latents)

    return

if __name__ == "__main__":
    src_path = "E:/Annonimized_CTPA_Xray_pairs/"
    dst_path = "D:/XRayCTPA/CTPA_256/"

    preprocess_ctpa_directory(src_path, dst_path)


