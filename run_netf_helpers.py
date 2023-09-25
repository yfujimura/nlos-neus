# This code is partially borrowed from NeTF.
# https://github.com/zeromakerplus/NeTF_public

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import scipy.io
from scipy import linalg
from scipy import signal
import multiprocessing
import numpy.matlib
import os

device = "cuda"
torch.autograd.set_detect_anomaly(True)


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
        
def save_model(out_dir, model, epoch, m, args, model_name, print_path=True):
    model_name = os.path.join(out_dir, '{}_epoch'.format(model_name) + str(epoch) + 'm' + str(m) + '.pt')
    params = model.state_dict()
    torch.save(params, model_name, pickle_protocol=4)
    if print_path:
        print("save as " + model_name)
        
        
def load_model(model, model_name, model_dir, epoch, m):
    params = torch.load(os.path.join(model_dir, '{}_epoch'.format(model_name) + str(epoch) + 'm' + str(m) + '.pt'))
    model.load_state_dict(params)
        
        
def _filter_transient(transient, kernel_size=31, mode="mean", std=2):
    if mode == "mean":
        kernel = np.ones(kernel_size)
    elif mode == "gaussian":
        kernel = signal.windows.gaussian(kernel_size, std)
    kernel = kernel / np.sum(kernel)
    filtered = signal.convolve(transient, kernel, mode="same")
            
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
        
    
def shuffle_data(nlos_data, camera_grid_positions):
    L, M, N = nlos_data.shape
    nlos_data = nlos_data.reshape(L,-1) # bins x (N_grid*N_grid)
    camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device) # 3 x (N_grid*N_grid)
    index = torch.linspace(0, M * N - 1, M * N).reshape(1, -1).float().to(device)
    full_data = torch.cat((nlos_data, camera_grid_positions, index), axis = 0)
    full_data = full_data[:,torch.randperm(full_data.size(1))]
    nlos_data = full_data[0:L,:].view(L,M,N)
    camera_grid_positions = full_data[L:-1,:].cpu().numpy()
    index = full_data[-1,:].cpu().numpy().astype(np.int)
    del full_data
    return nlos_data, camera_grid_positions, index


def spherical_sampling(L, 
                       camera_grid_positions, 
                       num_sampling_points, 
                       volume_position, volume_size, 
                       c, deltaT, 
                       start, end):
    [x0,y0,z0] = [camera_grid_positions[0],camera_grid_positions[1],camera_grid_positions[2]]

    box_point = volume_box_point(volume_position, volume_size) # return turn the coordinated of 8 points of the bounding box
    box_point[:,0] = box_point[:,0] - x0
    box_point[:,1] = box_point[:,1] - y0
    box_point[:,2] = box_point[:,2] - z0 # shift to fit the origin
    sphere_box_point = cartesian2spherical(box_point)
    theta_min = np.min(sphere_box_point[:,1]) - 0
    theta_max = np.max(sphere_box_point[:,1]) + 0
    phi_min = np.min(sphere_box_point[:,2]) - 0
    phi_max = np.max(sphere_box_point[:,2]) + 0
    theta = torch.linspace(theta_min, theta_max , num_sampling_points).float()
    phi = torch.linspace(phi_min, phi_max, num_sampling_points).float()
    dtheta = (theta_max - theta_min) / num_sampling_points
    dphi = (phi_max - phi_min) / num_sampling_points
   
    # Cartesian Coordinate System is refered to Zaragoza dataset: https://graphics.unizar.es/nlos_dataset.html
    # Spherical Coordinate System is refered to Wikipedia and ISO convention: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # theta: [0,pi]
    # phi: [-pi,pi], but we only use [-pi,0]

    r_min = start * c * deltaT
    r_max = end * c * deltaT

    num_r = math.ceil((r_max - r_min) / (c * deltaT))
    r = torch.linspace(r_min, r_max , num_r).float()

    I1 = r_min / (c * deltaT)
    I2 = r_max / (c * deltaT)

    I1 = math.floor(I1)
    I2 = math.ceil(I2)
    I0 = r.shape[0]

    grid = torch.stack(torch.meshgrid(r, theta, phi),axis = -1)

    spherical = grid.reshape([-1,3])
    cartesian = spherical2cartesian_torch(spherical)
    cartesian = cartesian + torch.tensor([x0,y0,z0])
    cartesian = torch.cat((cartesian, spherical[:,1:3]), axis = 1)
    return cartesian, I1, I2, I0, dtheta, dphi, theta_min, theta_max, phi_min, phi_max  


def volume_box_point(volume_position, volume_size):
    # format: volume_position: 3, vector  volume_size: scalar
    # output: box: 8 x 3
    [xv, yv, zv] = [volume_position[0], volume_position[1], volume_position[2]]
    # xv, yv, zv is the center of the volume
    x = np.array([xv - volume_size / 2, xv - volume_size / 2, xv - volume_size / 2, xv - volume_size / 2, xv + volume_size / 2, xv + volume_size / 2, xv + volume_size / 2, xv + volume_size / 2])
    y = np.array([yv - volume_size / 2, yv - volume_size / 2, yv + volume_size / 2, yv + volume_size / 2, yv - volume_size / 2, yv - volume_size / 2, yv + volume_size / 2, yv + volume_size / 2])
    z = np.array([zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2])
    box = np.stack((x, y, z),axis = 1)
    return box

def cartesian2spherical(pt):
    # cartesian to spherical coordinates
    # input： pt N x 3 ndarray

    spherical_pt = np.zeros(pt.shape)
    spherical_pt[:,0] = np.sqrt(np.sum(pt ** 2,axis=1))
    spherical_pt[:,1] = np.arccos(pt[:,2] / spherical_pt[:,0])
    phi_yplus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] >= 0)
    phi_yplus = phi_yplus + (phi_yplus < 0).astype(np.int) * (np.pi)
    phi_yminus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] < 0)
    phi_yminus = phi_yminus + (phi_yminus > 0).astype(np.int) * (-np.pi)
    spherical_pt[:,2] = phi_yminus + phi_yplus
    return spherical_pt

def spherical2cartesian(pt):
    # spherical to cartesian coordinates
    # input: pt N x 3 ndarray

    cartesian_pt = np.zeros(pt.shape)
    cartesian_pt[:,0] = pt[:,0] * np.sin(pt[:,1]) * np.cos(pt[:,2])
    cartesian_pt[:,1] = pt[:,0] * np.sin(pt[:,1]) * np.sin(pt[:,2])
    cartesian_pt[:,2] = pt[:,0] * np.cos(pt[:,1])

    return cartesian_pt

def spherical2cartesian_torch(pt):
    # spherical to cartesian coordinates
    # input： pt N x 3 tensor

    cartesian_pt = torch.zeros(pt.shape)
    cartesian_pt[:,0] = pt[:,0] * torch.sin(pt[:,1]) * torch.cos(pt[:,2])
    cartesian_pt[:,1] = pt[:,0] * torch.sin(pt[:,1]) * torch.sin(pt[:,2])
    cartesian_pt[:,2] = pt[:,0] * torch.cos(pt[:,1])

    return cartesian_pt


def compute_histogram_loss(args,
                           M,m,N,n,j,L,
                           criterion,
                           sdf_network, color_network, deviation_network, background_network,
                           camera_grid_positions,
                           nlos_data,
                           volume_position, volume_size, c, deltaT):
    camera_grid_position = torch.from_numpy(camera_grid_positions[:, m * N + n + j]).to(device)
    
    with torch.no_grad():
        [input_points, I1, I2, I0, dtheta, dphi, theta_min, theta_max, phi_min, phi_max] = spherical_sampling(L, 
                                                                                                              camera_grid_positions[:,m * N + n + j],
                                                                                                              args.num_sampling_points, 
                                                                                                              volume_position, volume_size, 
                                                                                                              c, deltaT,
                                                                                                              args.start, args.end)
    
    input_points_reshaped = input_points[:,0:3].reshape(I0,args.num_sampling_points,args.num_sampling_points,3)
    cartesian_directions = input_points_reshaped[0] - input_points_reshaped[1]
    cartesian_directions = cartesian_directions / torch.norm(cartesian_directions, dim=2,keepdim=True)
    cartesian_directions = cartesian_directions.reshape(1, args.num_sampling_points,args.num_sampling_points, 3)
    cartesian_directions = cartesian_directions.expand(I0,args.num_sampling_points,args.num_sampling_points,3)
    cartesian_directions = cartesian_directions.reshape(-1,3)
    
    # run models 
    network_res = sdf_network(input_points[:,0:3])
    sdf = network_res[:,:1]
    feature_vector = network_res[:,1:]
    sampled_color = color_network(input_points[:,0:3],
                                  None, 
                                  cartesian_directions, 
                                  feature_vector)[:,:1] 
    sampled_color = F.softplus(sampled_color, beta=100)
    inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6) # 1 x 1
    inv_s = inv_s.expand(sdf.shape[0], 1)
    
    # sdf => density => weight
    density = inv_s * torch.sigmoid(-sdf * inv_s)
    density = torch.reshape(density, (I0, args.num_sampling_points**2))
    alpha = 1.0 - torch.exp(-density*c*deltaT)
    weights = alpha * torch.cumprod(torch.cat([torch.ones([1, alpha.shape[1]]), 1. - alpha + 1e-7], 0), 0)[:-1, :]

    sampled_color = torch.reshape(sampled_color, (I0, args.num_sampling_points**2))
    network_res = weights * sampled_color
    
    pdf = network_res / (torch.sum(network_res, dim=1, keepdim=True) + 1e-8)
    cdf = torch.cumsum(pdf, -1)
        
    # entropy loss
    if args.weight_en != 0:
        cum_weights = torch.sum(weights, dim=0)
        cum_weights = torch.clamp(cum_weights, 1e-7, 1-1e-7)
        loss_en = torch.mean(-cum_weights * torch.log2(cum_weights) - (1 - cum_weights) * torch.log2(1-cum_weights))
    else:
        loss_en = 0.
    
    # attenuation factor
    with torch.no_grad():
        dists = (torch.linspace(I1, I2, I0) * deltaT * c).float().to(device)
        dists = dists.reshape(-1, 1)
        dists = dists.repeat(1, args.num_sampling_points ** 2)
        theta = input_points.reshape(-1, args.num_sampling_points ** 2, 5)[:,:,3]
    network_res = network_res / (dists ** 2) * torch.sin(theta)

    # histogram generation
    pred_histogram = torch.sum(network_res, axis = 1)
    pred_histogram = pred_histogram * dtheta * dphi
    
    nlos_histogram = nlos_data[I1:(I1 + I0), m, n + j] * args.gt_times
    
        
    # render background
    if args.render_background == 1:
        input_pts = camera_grid_position[[0,2]]
        input_pts = input_pts[None,:].expand(I0,2)
        times = torch.linspace(0,1,I0).unsqueeze(1)
        background_histogram = background_network(input_pts, times).squeeze()
        background_histogram = F.softplus(background_histogram, beta=100)
        background_scale = (torch.sum(nlos_histogram) - torch.sum(pred_histogram)) / (torch.sum(background_histogram) + 1e-8)
        background_histogram = background_scale * background_histogram
        pred_histogram = pred_histogram + background_histogram
    else:
        background_histogram = None
        
        
    # zero-sdf loss
    if args.weight_z > 0:
        sdf = sdf.reshape(I0, args.num_sampling_points ** 2)
        udf = torch.abs(sdf)
        temporal_mask = torch.zeros_like(nlos_data[I1:(I1 + I0), m, n + j])
        if args.render_background == 0:
            temporal_mask[nlos_data[I1:(I1 + I0), m, n + j] > args.transient_threshold] = 1
        else:
            temporal_mask[(nlos_data[I1:(I1 + I0), m, n + j] - background_histogram / args.gt_times) > args.transient_threshold] = 1
        N_samples = args.num_sampling_zero_pts
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
        inds = torch.searchsorted(cdf, u, right=False) 
        inds = inds.reshape((-1,))
        inds[inds>=args.num_sampling_points ** 2] = args.num_sampling_points ** 2-1
        bin_inds = torch.tensor([[k]*N_samples for k in range(I0)]).to("cuda")
        bin_inds = bin_inds.reshape((-1,))
        target_udf = udf[bin_inds, inds]
        target_udf = target_udf.reshape(I0, N_samples)
        loss_z = torch.sum(target_udf*temporal_mask[:,None]) / (torch.sum(temporal_mask)*N_samples + 1e-8)
    else:
        loss_z = 0.

        
        
    
    loss_h = criterion(pred_histogram, nlos_histogram)

    
    return loss_h, loss_z, loss_en, nlos_histogram, pred_histogram, background_histogram


def compute_eikonal_loss(sdf_network, volume_position, volume_size, N_samples=4096):
    min_vals = torch.from_numpy((volume_position - volume_size*0.5).astype(np.float32)).to(device)
    max_vals = torch.from_numpy((volume_position + volume_size*0.5).astype(np.float32)).to(device)
    pts = min_vals[None,:] + torch.rand(N_samples,3) * (max_vals - min_vals)
    
    _, gradients = sdf_network.gradient(pts, activation=None)
    
    gradients = gradients.squeeze() # N_samples x 3
    gradient_error = torch.mean((torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2)
    
    return gradient_error

def compute_freespace_loss(sdf_network, freespace_sdf, N_samples=4096, lower_bound=1):
    with torch.no_grad():
        batch_indices = np.random.choice(freespace_sdf.shape[0], N_samples)
        freespace_sdf_batch = freespace_sdf[batch_indices]
        pts = freespace_sdf_batch[:,0:3]
        sdf = freespace_sdf_batch[:,3:4]
    
    sdf_pred = sdf_network(pts)[:,:1]
    if lower_bound == 1:
        loss = torch.mean(torch.relu(sdf - sdf_pred))
    else:
        loss = torch.nn.L1Loss()(sdf, sdf_pred)
    
    
    return loss

    
