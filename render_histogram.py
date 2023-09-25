# This is the code for rendering histograms after training.
# usage:
#   python render_histogram.py --config configs/test/zaragoza_bunny.txt

import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import scipy.io
from math import ceil

from run_netf_helpers import *
from load_nlos import *
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


def train():
    parser = config_parser()
    args = parser.parse_args()
    seed = args.rng
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)

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
    
    create_dir("{}_histogram".format(scene))
        
    nlos_data = torch.Tensor(nlos_data).to(device) 
    L,M,N = nlos_data.shape
    
    # create output dir
    dir_param_list = [scene, 
                      args.weight_h, args.weight_ei, args.weight_z, args.weight_f, args.weight_en, 
                      args.transient_threshold, 
                      args.num_sampling_zero_pts,
                      args.render_background, 
                     ]
    out_dir = os.path.join("out", "_".join([str(param) for param in dir_param_list]))
 
    # create models 
    if args.geometric_init == 1:
        geometric_init = True
    elif args.geometric_init == 0:
        geometric_init = False
    sdf_network = SDFNetwork(d_in=3,
                         d_out=257,
                         d_hidden=256,
                         n_layers=8,
                         skip_in=[4],
                         multires=6,
                         bias=0.5,
                         scale=1.0,
                         geometric_init=geometric_init,
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
    if args.render_background == 1:
        background_network = BackgroundNetwork(d_out=1,
                                               d_hidden=16,
                                               n_layers=3,
                                               multires_pts=3, multires_time=3, 
                                               squeeze_out=False)
        load_model(background_network, "background", out_dir, args.test_epoch, args.test_m)
    else:
        background_network = None
        
    criterion = torch.nn.MSELoss(reduction='mean')
    
    # shuffle data
    nlos_data, camera_grid_positions, index = shuffle_data(nlos_data, camera_grid_positions)
    current_nlos_data = nlos_data
    current_camera_grid_positions = camera_grid_positions
    
    with torch.no_grad():
        total_loss = test(args, criterion,
                          sdf_network, color_network, deviation_network, background_network, 
                          current_nlos_data, current_camera_grid_positions, index, 
                          volume_position, volume_size, c, deltaT, out_dir = "{}_histogram".format(scene))
        
        
def test(args, criterion,
         sdf_network, color_network, deviation_network, background_network, 
         current_nlos_data, current_camera_grid_positions, index, 
         volume_position, volume_size, c, deltaT,
         out_dir="histogram"):
    L,M,N = current_nlos_data.shape
    total_loss = torch.zeros(M * N)
    
    print("run test:")
    for m in tqdm(range(0, M, 1)):
        # batchsize is 1
        for n in range(0, N, 1):
            # minibatch
            for j in range(0, 1, 1):
                loss, _, _, gt_histogram, pred_histogram, background_histogram = compute_histogram_loss(args,
                                                                                                        M,m,N,n,j,L,
                                                                                                        criterion,
                                                                                                        sdf_network, color_network, deviation_network, background_network,
                                                                                                        current_camera_grid_positions,
                                                                                                        current_nlos_data,
                                                                                                        volume_position, volume_size, c, deltaT)
                
                total_loss[index[m * N + n]] = loss.item() / (torch.mean(gt_histogram) + 1e-8)
                np.save(os.path.join(out_dir, "gt_hist_{}".format(index[m * N + n])), gt_histogram.cpu().numpy())
                np.save(os.path.join(out_dir, "pred_hist_{}".format(index[m * N + n])), pred_histogram.cpu().numpy())
                if args.render_background == 1:
                    np.save(os.path.join(out_dir, "background_hist_{}".format(index[m * N + n])), background_histogram.cpu().numpy())
        exit()
                
    return total_loss

            
                            
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
