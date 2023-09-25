# This is the code for rendering directional albedos after training.
# usage:
#   python render_albedo.py --config configs/test/zaragoza_bunny.txt

import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy import signal
from scipy.spatial.transform import Rotation as R
import scipy.io
import matplotlib.pyplot as plt
from run_netf_helpers import *

from load_nlos import *
from math import ceil

import sys

import cv2
import mcubes

from fields import *
    
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--nlos_file", type=str, default=None, 
                        help='input data path')
    parser.add_argument("--dataset_type", type=str, default='zaragoza256', 
                        help='options: zaragoza256 / fk')

    # NeTF arguments
    parser.add_argument("--num_sampling_points", type=int, default=16, 
                        help='number of sampling points in one direction, so the number of all sampling points is the square of this value')
    parser.add_argument("--histogram_batchsize", type=int, default=1, 
                        help='the batchsize of histogram')
    parser.add_argument("--start", type=int, default=100, 
                        help='the start point of histogram')
    parser.add_argument("--end", type=int, default=300, 
                        help='the end point of histogram')
    parser.add_argument("--gt_times", type=float, default=100, 
                        help='scaling factor of histogram')
    parser.add_argument("--num_epochs", type=int, default=10, 
                        help='number of training epochs')
    parser.add_argument("--rng", type=int, default=1, 
                        help='random seed')
    
    # our options
    parser.add_argument("--init_lr", type=float, default=1e-4,
                        help="initial learning rate")
    parser.add_argument("--weight_h", type=float, default=1.,
                        help="weight for histogram loss")
    parser.add_argument("--weight_ei", type=float, default=1e-1,
                        help="weight for eikonal loss")
    parser.add_argument("--weight_z", type=float, default=0.,
                        help="weight for zero-sdf loss")
    parser.add_argument("--weight_f", type=float, default=0.,
                        help="weight for freespace loss")
    parser.add_argument("--weight_en", type=float, default=0.,
                        help="weight for entropy loss")
    parser.add_argument("--save_m", type=int, default=16, 
                        help="model save interval")
    parser.add_argument("--transient_threshold", type=float, default=0, 
                        help="threshold value for transient mask")
    parser.add_argument("--geometric_init", type=int, default=1, 
                        help="require geometric initialization (1) or not (0)")
    parser.add_argument("--num_sampling_zero_pts", type=int, default=32, 
                        help="number of sampling points on each sphere for zero-sdf loss")
    parser.add_argument("--num_sampling_eik_pts", type=int, default=4096, 
                        help="number of sampling points for eikonal loss")
    parser.add_argument("--num_sampling_lb_pts", type=int, default=4096, 
                        help="number of sampling points for freespace loss")
    parser.add_argument("--render_background", type=int, default=0, 
                        help="render background (1) or not (0)")
    
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--test_m", type=int, default=0)
    parser.add_argument("--test_volume_size", type=int, default=256)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--require_mask", type=int, default=1)
    parser.add_argument("--remove_mesh_th", type=float, default=0.1)
    
    return parser

def test_volume():
    parser = config_parser()
    args = parser.parse_args()
    seed = args.rng
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)
    
    create_dir("recon")
    
    # load data
    if args.dataset_type == "zaragoza256":
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_zaragoza256_data(args.nlos_file)
    elif args.dataset_type == "fk":
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_fk_data(args.nlos_file)
    # target volume is centered at the origin
    camera_grid_positions = camera_grid_positions - volume_position[:,None]
    volume_position = np.zeros(3)
    vmin = volume_position - volume_size / 2
    vmax = volume_position + volume_size / 2
    scene = os.path.splitext(os.path.basename(args.config))[0]
    
    dir_param_list = [scene, 
                      args.weight_h, args.weight_ei, args.weight_z, args.weight_f, args.weight_en,
                      args.transient_threshold, 
                      args.num_sampling_zero_pts,
                      args.render_background,
                     ]
    out_dir = os.path.join("out", "_".join([str(param) for param in dir_param_list]))
    
    
    # create models 
    sdf_network = SDFNetwork(d_in=3,
                         d_out=257,
                         d_hidden=256,
                         n_layers=8,
                         skip_in=[4],
                         multires=6,
                         bias=0.5,
                         scale=1.0,
                         geometric_init=True,
                         weight_norm=True,
                         inside_outside=False)
    color_network = RenderingNetwork(d_feature=256,
                                     mode="no_normal",
                                     d_in=6,
                                     d_out=1,
                                     d_hidden=256,
                                     n_layers=4,
                                     weight_norm=True,
                                     multires_view=4,
                                     squeeze_out=False)
    deviation_network = SingleVarianceNetwork(init_val=0.3)
    load_model(sdf_network, "sdf", out_dir, args.test_epoch, args.test_m)
    load_model(color_network, "color", out_dir, args.test_epoch, args.test_m)
    load_model(deviation_network, "deviation", out_dir, args.test_epoch, args.test_m)

    unit_distance = (volume_size) / (args.test_volume_size - 1) 
    xv = yv = zv = np.linspace(-volume_size / 2, volume_size / 2, args.test_volume_size) 
    
    coords = np.stack(np.meshgrid(xv, yv, zv),-1) # coords
    coords = coords.transpose([1,0,2,3])
    coords = coords.reshape([-1,3])
    
    camera_grid_positions = np.reshape(camera_grid_positions, (3, 256, 256))
    
    if args.dataset_type == "zaragoza256":
        rmats = [R.from_rotvec(np.array([0, 0, 0])).as_matrix(),
                 R.from_rotvec(np.array([0, 0, np.pi/4])).as_matrix(),
                 R.from_rotvec(np.array([0, 0, -np.pi/4])).as_matrix(),
                ]
    elif args.dataset_type == "fk":
        rmats = [R.from_rotvec(np.array([0, 0, 0])).as_matrix(),
                 R.from_rotvec(np.array([np.pi/6, 0, 0])).as_matrix(),
                 R.from_rotvec(np.array([-np.pi/6, 0, 0])).as_matrix(),
                ]
    
    with torch.no_grad():
        num_pts = args.test_volume_size**3
        num_steps = 32
        num_batch = num_pts // num_steps
        sdfs = torch.zeros(len(rmats), num_pts,1)
        sampled_colors = torch.zeros(len(rmats), num_pts,1)
        
        print("compute sdf volume from {} views ...".format(len(rmats)))
        for j, rmat in enumerate(rmats):
            coords_r = np.dot(coords[:,0:3], rmat.T)
            test_input = torch.from_numpy(coords_r).float().to(device)
            
            view_dir = test_input.reshape([args.test_volume_size, args.test_volume_size, args.test_volume_size, 3])
            view_dir = view_dir[0,1,0] - view_dir[0,0,0]
            view_dir = view_dir / torch.norm(view_dir)
            
            for i in tqdm(range(num_steps)):
                out_sdf = sdf_network(test_input[i*num_batch:(i+1)*num_batch])
                sdfs[j, i*num_batch:(i+1)*num_batch] = out_sdf[:,:1]

                out_color = color_network(test_input[i*num_batch:(i+1)*num_batch],
                                          None, 
                                          view_dir[None,:].expand(num_batch,3), 
                                          out_sdf[:,1:])[:,:1] 
                out_color = F.softplus(out_color, beta=100)
                sampled_colors[j,i*num_batch:(i+1)*num_batch] = out_color
                
            # reflectance outside the target volume is zero
            mask = torch.ones(num_pts)
            mask[test_input[:,0] < -volume_size / 2] = 0
            mask[test_input[:,0] > volume_size / 2] = 0
            mask[test_input[:,1] < -volume_size / 2] = 0
            mask[test_input[:,1] > volume_size / 2] = 0
            mask[test_input[:,2] < -volume_size / 2] = 0
            mask[test_input[:,2] > volume_size / 2] = 0
            sampled_colors[j] = sampled_colors[j] * mask[:,None]

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6) 
        inv_s = inv_s.expand(len(rmats), num_pts, 1)
        
        densities = inv_s * torch.sigmoid(-sdfs * inv_s)
        densities = torch.reshape(densities, (len(rmats), args.test_volume_size,args.test_volume_size,args.test_volume_size))
        alphas = 1.0 - torch.exp(-densities*unit_distance)
        dists = np.mean(camera_grid_positions[1]) - volume_size/2 + np.linspace(volume_size, 0, args.test_volume_size)
        
        color_volume = sampled_colors[:,:,0].view(len(rmats), args.test_volume_size,args.test_volume_size,args.test_volume_size)
        color_volume = color_volume.cpu().detach().numpy()
        
        for j in range(len(rmats)):
            trans = torch.ones_like(alphas[j])
            for i in range(args.test_volume_size-2, -1, -1):
                trans[:,i,:] = torch.prod(1 - alphas[j,:,i+1:args.test_volume_size,:] + 1e-7, dim=1)     
            weights = alphas[j] * trans
            weights = weights.cpu().numpy()
            albedo = weights * color_volume[j] * unit_distance / dists[None,:,None]**2
            albedo = np.sum(albedo, axis=1)
            np.save("recon/{}_albedo{}".format(scene,j), albedo)
            
            tmp = weights * color_volume[j] * unit_distance
            #tmp = weights * color_volume[j] * unit_distance / dists[None,:,None]**2
            tmp = np.sum(tmp, axis=1)
            th, _ = cv2.threshold((tmp * 65535. / np.max(tmp)).astype(np.uint16).reshape(-1,1), 0, 65535, cv2.THRESH_OTSU)
            th = th * np.max(tmp)/65535.
            mask = np.zeros_like(tmp)
            mask[tmp >= th] = 1
            np.save("recon/{}_mask{}".format(scene,j), mask)

        



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    test_volume()

