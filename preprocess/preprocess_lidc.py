import os
import numpy as np
import matplotlib.pyplot as plt
plt.set_cmap('gray')
#import nibabel as nib
import pylidc as pl
import scipy
from scipy.ndimage import zoom
import warnings
from params import *
from diffusers import AutoencoderKL
import torch
from lungmask import mask
import SimpleITK as sitk

CONTRAST_HU_MIN = -1200.     # -1000.# Min value for loading contrast
CONTRAST_HU_MAX = 600.      # Max value for loading contrast

def normalize(img):
    
    img = img.astype(np.float32)
    img = (img - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN) 
    img = np.clip(img, 0., 1.) * 2 -1
    return img

def resample_volume(volume, current_spacing, new_spacing):
    max_size  = 4096
    resize_factor = np.array(current_spacing) / new_spacing
    current_size = volume.shape
    new_shape = np.round(current_size * resize_factor)
    new_size = tuple([min(size, max_size) for size in new_shape])
    resampled_volume = zoom(volume, np.array(new_size) / current_size, order=1)
    return resampled_volume

# Resize as required
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

def preprocess_ct(scan):
    vol = scan.to_volume()
    #print(vol.shape) # (dim, dim, depth)

    vol_cropped, lungs = get_lungs_voi(vol)
    print("Cropping scan to lung VOI:   ", vol_cropped.shape)
    z, x,y = vol_cropped.shape
    if (z and x and y) == 0:
        print(" failed to compute voi : compute manually")
        return 0,0

    # Resample
    vol_resampled = resample_volume(vol_cropped, scan.slice_spacing, [0.703125, 0.703125, 1.25])#[1,1,1])

    vol_resize = resize_volume(vol_resampled, [256,256,128])

    print("after crop = ", vol_resize.shape)
    vol_norm = normalize(vol_resize)
    return vol_norm

def make_rgb(volume):
    """Tile a NumPy array to make sure it has 3 channels."""
    z, c, h, w = volume.shape

    tiling_shape = [1]*(len(volume.shape))
    tiling_shape[1] = 3
    np_vol = torch.tile(volume, tiling_shape)
    return np_vol

def encode_vae(ct_np, vae):

    device = "cuda"
    weight_dtype =  torch.float32
    ct = torch.from_numpy(ct_np).unsqueeze(0).unsqueeze(0)
    vae.eval()

    latents = []
    with torch.no_grad():
        for i in range(ct.shape[4]):
            slice = make_rgb(ct[:,:,:,:,i])
            latent = vae.encode(slice.to(device, dtype=weight_dtype)).latent_dist.sample()
            latents.append(latent)
        latents = torch.stack(latents, dim=4).squeeze()

        latents = latents.detach().cpu().numpy()
        latents = latents * 0.18215

    return latents

def convert_dicom_to_numpy(pid, src_path, dst_path_vae, dst_path_py, vae):

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    vol = preprocess_ct(scan)
    print(vol.shape)
    #sitk.WriteImage(sitk.GetImageFromArray(vol), dst_path + '.nii')

    latents = encode_vae(vol, vae)
    print(latents.shape)

    output_file_vae = dst_path_vae + '.npy'
    output_file_py = dst_path_py + '.npy'
    print(output_file_vae)

    np.save(output_file_vae, latents)
    np.save(output_file_py, vol)
    #sitk.WriteImage(sitk.GetImageFromArray(vol), output_file_py + '.nii')

    return

def convert_dicom_directories_to_numpy(src_path, dst_path_vae, dst_path_py, vae):

    for pid in os.listdir(src_path):
        scan_src = src_path + pid
        scan_dst_vae = dst_path_vae + pid
        scan_dst_py = dst_path_py + pid
        print(" scan = ", pid)

        if os.path.isfile(scan_dst_vae + '.npy'):
            print("File exists - ", scan_dst_vae + '.npy')
        else:
            convert_dicom_to_numpy(pid, scan_src, scan_dst_vae, scan_dst_py, vae)

    return

if __name__ == "__main__":

    src_path = "../datasets/LIDC/LIDC-IDRI/"
    dst_path_vae = "../datasets/LIDC/lidc_vae_256/"
    dst_path = "D:/LIDC/lidc_256/"

    device = DEVICE
    weight_dtype = torch.float32

    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    vae = AutoencoderKL.from_single_file(url)
    vae.to(device, dtype=weight_dtype)

    convert_dicom_directories_to_numpy(src_path, dst_path_vae, dst_path, vae)

