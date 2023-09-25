# This is the code for space carving based on the geometry of the first-returning photons.
# Before training, please run this code to create sdf lower bounds.
# usage:
#   python space_carving.py --scene zaragoza_bunny

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
import tqdm
import torch
import math
from scipy import signal
import scipy.io as scio
from scipy.optimize import curve_fit  
from scipy.ndimage import gaussian_filter1d
import argparse

from load_nlos import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--scene", default="zaragoza_bunny", 
                        help="zaragoza_bunny | zaragoza_lucy | zaragoza_indonesian | fk_statue | fk_dragon | fk_bike")
    parser.add_argument("--target_volume_size", type=int, default=128)
    parser.add_argument("--ratio", type=int, default=0.99)
    args = parser.parse_args()
    return args


def _filter_transient(transient, kernel_size=31, mode="mean", std=2):
    if mode == "mean":
        kernel = np.ones(kernel_size)
        kernel = kernel / np.sum(kernel)
        filtered = signal.convolve(transient, kernel, mode="same")
    elif mode == "gaussian":
        kernel = signal.windows.gaussian(kernel_size, std)
        kernel = kernel / np.sum(kernel)
        filtered = signal.convolve(transient, kernel, mode="same")
    elif mode == "median":
        filtered = signal.medfilt(transient, kernel_size)
            
    return filtered

def filter_transient(transient, kernel_size=31, mode="mean", std=2, bias_start=300, threshold=1.5):
    bins, height, width = transient.shape
    filtered = np.zeros_like(transient)
    mean_bias = 0
    
    for y in range(height):
        for x in range(width):
            _filtered = _filter_transient(transient[:,y,x], kernel_size, mode, std)
            bias = np.mean(_filtered[bias_start:])
            _filtered = _filtered - bias
            _filtered[_filtered < 0] = 0
            filtered[:,y,x] = _filtered
            mean_bias += bias
    mean_bias /= (height*width)
    
    filtered[filtered <= threshold] = 0
            
    return filtered, mean_bias


def detect_first_bounces(transient, threshold=1e-5):
    bins, height, width = transient.shape
    
    first_bounces = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            if np.sum(transient[:,y,x]) != 0:
                for b in range(1, bins, 1):
                    if transient[b,y,x] - transient[b-1,y,x]  > threshold:
                        first_bounces[y,x] = b
                        break
    return first_bounces


def space_carving(args):
    # load data
    if args.scene == "zaragoza_bunny":
        nlos_file = "data/zaragozadataset/zaragoza256_preprocessed.mat"
        dataset_type = "zaragoza256"
        start = 0
        threshold = 1e-5
    if args.scene == "zaragoza_lucy":
        nlos_file = "data/zaragozadataset/zaragoza_lucy_256_1m_preprocessed.mat"
        dataset_type = "zaragoza256"
        start = 250
        threshold = 1e-5
    if args.scene == "zaragoza_indonesian":
        nlos_file = "data/zaragozadataset/zaragoza_indonesian_256_1m_preprocessed.mat"
        dataset_type = "zaragoza256"
        start = 250
        threshold = 1e-5
    if args.scene == "zaragoza_semioccluded":
        nlos_file = "data/zaragozadataset/zaragoza_semioccluded_256_preprocessed.mat"
        dataset_type = "zaragoza256"
        start = 0
        threshold = 1e-5
    if args.scene == "fk_statue":
        nlos_file = "data/fk_statue_meas_180_min_256_preprocessed.mat"
        dataset_type = "fk"
        start = 0
        threshold = 0.
    if args.scene == "fk_dragon":
        nlos_file = "data/fk_dragon_meas_180_min_256_preprocessed.mat"
        dataset_type = "fk"
        start = 0
        threshold = 0.
    if args.scene == "fk_bike":
        nlos_file = "data/fk_bike_meas_180_min_256_preprocessed.mat"
        dataset_type = "fk"
        start = 0
        threshold = 0.
        
        
    if dataset_type == 'zaragoza256':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_zaragoza256_data(nlos_file)
    elif dataset_type == 'fk':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_fk_data(nlos_file)
        # for detecting first bounces in fk-dataset, transient histogram is smoothed and thresholded
        nlos_data, bias = filter_transient(nlos_data, kernel_size=7, std=1, bias_start=400, threshold=3, mode="gaussian")
    camera_grid_positions = camera_grid_positions - volume_position[:,None]
    camera_grid_positions = torch.from_numpy(camera_grid_positions).to(device)
    volume_position = np.zeros((1,3))
    vmin = volume_position - volume_size / 2
    vmax = volume_position + volume_size / 2
            
        
        
    print("detect first bounces ...")
    radiuses = start + detect_first_bounces(nlos_data[start:], threshold=threshold)
    radiuses = radiuses * c * deltaT
    radiuses = radiuses.reshape(-1,)
    
    unit_distance = volume_size / (args.target_volume_size - 1) 
    x_coords = y_coords = z_coords = np.linspace(-volume_size / 2, volume_size / 2, args.target_volume_size) 
    
    coords = np.stack(np.meshgrid(x_coords, y_coords, z_coords),-1) 
    coords = coords.transpose([1,0,2,3])
    coords = coords.reshape([-1,3]) # target_volume_size**3 x 3
    
    coords = torch.from_numpy(coords.astype(np.float32)).to(device)
    votes = torch.zeros(coords.shape[0])
    
    print("space carving ...")
    total_votes = 0
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, camera_grid_positions.shape[1], 1)):
            if radiuses[i] > 0:
                total_votes += 1
                
                pt0 = camera_grid_positions[:,i]
                
                v = coords - pt0[None,:]
                diffs = torch.norm(v, dim=1)
                mask = torch.ones_like(diffs)
                mask[diffs < radiuses[i]] = 0
                votes[mask > 0] = votes[mask > 0] + 1
    
    threshold = torch.max(votes).item()*args.ratio
    mask = torch.zeros_like(votes)
    mask[votes > threshold] = 1
    mask[votes <= threshold] = 0
    
    coords2 = coords[torch.nonzero(mask, as_tuple=True)[0]]
    print("result: {} voxels => {} voxels.".format(args.target_volume_size**3, coords2.shape[0]))
    
    voxel_inds = np.round(((coords2.to("cpu").numpy() - vmin[:,0]) / unit_distance)).astype(np.int32)
    voxelarray = np.zeros((args.target_volume_size,args.target_volume_size,args.target_volume_size))
    voxelarray[voxel_inds[:,0], voxel_inds[:,1], voxel_inds[:,2]] = 1
    freespace = 1 - voxelarray
    
    
    print("compute freespace sdf ...")
    coords_volume = torch.reshape(coords, (args.target_volume_size,args.target_volume_size,args.target_volume_size, 3))
    sdf_volume = torch.ones((args.target_volume_size,args.target_volume_size,args.target_volume_size))
    for z in tqdm.tqdm(range(args.target_volume_size)):
        for y in range(args.target_volume_size):
            for x in range(args.target_volume_size):
                if freespace[x,y,z] == 1:
                    coord = coords_volume[x,y,z]
                    sdf = torch.min(torch.norm(coords2 - coord[None,:], dim=1))
                    sdf_volume[x,y,z] = sdf
                    
    
    # save             
    coords_freespace = coords_volume[freespace==1]
    sdf_freespace = sdf_volume[freespace==1]
    coords_sdf = torch.cat((coords_freespace, sdf_freespace[:,None]), 1)
    np.save("freespace_sdf/{}_freespace_sdf".format(args.scene), coords_sdf.to("cpu").numpy())
    

if __name__ == "__main__":
    args = parse_args()
    space_carving(args)