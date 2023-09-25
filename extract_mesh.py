# This is the code for extracting mesh after training.
# If you use this code with the mask option true, 
# please run render_albedo.py beforehand because it generates an object mask.
# usage:
#   python extract_mesh.py --config configs/test/zaragoza_bunny.txt

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
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from run_netf_helpers import *
import open3d as o3d

from load_nlos import *
from math import ceil

import sys

import cv2
import mcubes

from fields import *
        
    

def save_pcd(filename, pts, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(filename, pcd)
    return pcd


def poisson_recon(filename, pcd, remove_mesh_th=0.1):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    #densities = np.array(densities)
    #th, _ = cv2.threshold((densities * 65535. / np.max(densities)).astype(np.uint16).reshape(-1,1), 0, 65535, cv2.THRESH_OTSU)
    #th = th * np.max(densities)/65535.
    #vertices_to_remove = densities < th
    vertices_to_remove = densities < np.quantile(densities, remove_mesh_th)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    o3d.io.write_triangle_mesh(filename, mesh)
    
    

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
    coords = coords.reshape([-1,3])
        
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
        
    view_dirs = []
    start_pts_list = []
    for rmat in rmats:
        coords_r = np.dot(coords[:,0:3], rmat.T)
        coords_r = coords_r.reshape([args.test_volume_size, args.test_volume_size, args.test_volume_size, 3])
        
        view_dir = coords_r[0,0,0] - coords_r[0,1,0]
        view_dir = torch.from_numpy(view_dir / np.linalg.norm(view_dir)).float().to(device)
        view_dirs.append(view_dir)
        
        start_pts = coords_r[:,-1,:]
        start_pts = torch.from_numpy(start_pts.reshape(-1,3)).float().to(device)
        start_pts_list.append(start_pts)
        
    
    
    if args.require_mask == 1:
        masks = []
        for i in range(len(view_dirs)):
            mask = np.load("recon/{}_mask{}.npy".format(scene,i))
            masks.append(mask.reshape((-1,)))
    
    
    pts_all = torch.zeros(0,3)
    with torch.no_grad():
        print("sphere tracing ...")
        print("num_views: {}, num iters: {}".format(len(view_dirs), args.num_iters))
        for j, (view_dir, start_pts) in enumerate(zip(view_dirs, start_pts_list)):
            curr_pts = start_pts
            if args.require_mask == 1:
                curr_pts = curr_pts[np.nonzero(masks[j])]
            for i in tqdm(range(args.num_iters)):
                sdf = sdf_network(curr_pts)[:,:1]
                curr_pts = curr_pts + view_dir * sdf
                mask = torch.ones(curr_pts.shape[0])
                mask[torch.any(curr_pts< -volume_size*0.5, dim=1)] = 0
                mask[torch.any(curr_pts> volume_size*0.5, dim=1)] = 0
                curr_pts = curr_pts[torch.nonzero(mask, as_tuple=True)]
                
            pts_all = torch.cat((pts_all, curr_pts), 0)
            
    pts_all.requires_grad = True
    _, normals = sdf_network.gradient(pts_all, activation=None)
    normals = normals.squeeze()
    normals = normals.to("cpu").detach().numpy()
        
    pts_all = pts_all.to("cpu").detach().numpy()
    if args.require_mask == 1:
        pcd = save_pcd(os.path.join("recon", "{}_pcd_wmask.ply".format(scene)), pts_all, normals)
    else:
        pcd = save_pcd(os.path.join("recon", "{}_pcd_womask.ply".format(scene)), pts_all, normals)
    
    print("poisson surface reconstruction ...")
    if args.require_mask == 1:
        poisson_recon(os.path.join("recon", "{}_mesh_wmask.ply".format(scene)), pcd, remove_mesh_th=args.remove_mesh_th)
    else:
        poisson_recon(os.path.join("recon", "{}_mesh_womask.ply".format(scene)), pcd, remove_mesh_th=args.remove_mesh_th)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    extract_mesh()

