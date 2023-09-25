# This is the code for rendering depth after training.
# usage:
#   python render_depth.py --config configs/test/zaragoza_bunny.txt --test_volume_size 207

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
import scipy.io
import matplotlib.pyplot as plt
from run_netf_helpers import *
import open3d as o3d

from load_nlos import *
from math import ceil

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
    parser.add_argument("--test_volume_size", type=int, default=207)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--require_mask", type=int, default=1)
    parser.add_argument("--remove_mesh_th", type=float, default=0.1)
    
    return parser


def extract_mesh():
    parser = config_parser()
    args = parser.parse_args()
    seed = args.rng
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)
    
    create_dir("recon")

    # Load data
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
    
    view_dir = np.array([0.,-1.,0.])
    view_dir = torch.from_numpy(view_dir).float().unsqueeze(0).to(device)
    
    start_pts = coords[:,-1,:]
    start_pts = torch.from_numpy(start_pts.reshape(-1,3)).float().to(device)
    
    pts_all = torch.zeros(0,3)
    with torch.no_grad():
        print("sphere tracing ...")
        print("num iters: {}".format(args.num_iters))
        curr_pts = start_pts
        mask = torch.ones(curr_pts.shape[0])
        for i in tqdm(range(args.num_iters)):
            sdf = sdf_network(curr_pts)[:,:1]
            curr_pts = curr_pts + mask[:,None] * view_dir * sdf
            mask[torch.any(curr_pts< -volume_size*0.5, dim=1)] = 0
            mask[torch.any(curr_pts> volume_size*0.5, dim=1)] = 0
            
    curr_pts.requires_grad = True
    _, normals = sdf_network.gradient(curr_pts, activation=None)
    normals = normals.squeeze() # N_samples x 3 
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
            
    curr_pts = curr_pts.reshape(args.test_volume_size, args.test_volume_size, 3)
    depth = np.mean(camera_grid_positions[1]) - curr_pts[:,:,1]
    np.save("recon/{}_depth".format(scene), depth.to("cpu").detach().numpy())
    
    normals = normals.reshape(args.test_volume_size, args.test_volume_size, 3)
    np.save("recon/{}_normal".format(scene), normals.to("cpu").detach().numpy())
            
           
        


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    extract_mesh()

