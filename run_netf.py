# This code is partially borrowed from NeTF.
# https://github.com/zeromakerplus/NeTF_public

# usage:
#   python run_netf.py --config configs/zaragoza_bunny.txt

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
        
    nlos_data = torch.Tensor(nlos_data).to(device) 
    L,M,N = nlos_data.shape
        
    # load freespace sdf
    freespace_sdf = np.load('freespace_sdf/{}_freespace_sdf.npy'.format(scene))
    freespace_sdf = torch.from_numpy(freespace_sdf.astype(np.float32)).to(device)
    
    # create output dir
    dir_param_list = [scene, 
                      args.weight_h, args.weight_ei, args.weight_z, args.weight_f, args.weight_en, 
                      args.transient_threshold, 
                      args.num_sampling_zero_pts,
                      args.render_background,
                     ]
    out_dir = os.path.join("out", "_".join([str(param) for param in dir_param_list]))
    create_dir("out")
    create_dir(out_dir)

    
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
    params_to_train = []
    params_to_train += list(sdf_network.parameters())
    params_to_train += list(color_network.parameters())
    params_to_train += list(deviation_network.parameters())
    if args.render_background == 1:
        background_network = BackgroundNetwork(d_out=1,
                                               d_hidden=16,
                                               n_layers=3,
                                               multires_pts=3, multires_time=3, 
                                               squeeze_out=False)
        params_to_train += list(background_network.parameters())
    else:
        background_network = None
    optimizer = torch.optim.Adam(params_to_train, lr=args.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion = torch.nn.MSELoss(reduction='mean')
    
    
    # shuffle data
    nlos_data, camera_grid_positions, index = shuffle_data(nlos_data, camera_grid_positions)
    current_nlos_data = nlos_data
    current_camera_grid_positions = camera_grid_positions
    
    
    # start training
    time0 = time.time()
    total_iter = 0
    for epoch in trange(args.num_epochs):
        print(epoch, '/', args.num_epochs)
        
        for m in range(0, M, 1):
            # save models
            if (m % args.save_m) == 0:
                save_model(out_dir, sdf_network, epoch, m, args, model_name="sdf", print_path=True)
                save_model(out_dir, color_network, epoch, m, args, model_name="color", print_path=True)
                save_model(out_dir, deviation_network, epoch, m, args, model_name="deviation", print_path=True)
                if args.render_background == 1:
                    save_model(out_dir, background_network, epoch, m, args, model_name="background", print_path=True)
                    

            for n in range(0, N, args.histogram_batchsize):
                # minibatch
                loss_batch = 0
                for j in range(0, args.histogram_batchsize, 1):
                    loss_h, loss_z, loss_en, _, _, _ = compute_histogram_loss(args,
                                                                          M,m,N,n,j,L,
                                                                          criterion,
                                                                          sdf_network, color_network, deviation_network, background_network,
                                                                          current_camera_grid_positions,
                                                                          current_nlos_data,
                                                                          volume_position, volume_size, c, deltaT)

                    if args.weight_ei != 0:
                        loss_ei = compute_eikonal_loss(sdf_network, volume_position, volume_size, N_samples=args.num_sampling_eik_pts)
                    else:
                        loss_ei = 0.


                    if args.weight_f != 0:
                        loss_f = compute_freespace_loss(sdf_network, freespace_sdf, N_samples=args.num_sampling_lb_pts, lower_bound=1)
                    else:
                        loss_f = 0.
                        

                    loss = args.weight_h * loss_h + args.weight_ei * loss_ei + args.weight_f * loss_f + args.weight_z * loss_z + args.weight_en * loss_en
                    loss_batch += loss 
                    
                loss_batch = loss_batch / args.histogram_batchsize
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()

                if (n % 16 == 0):
                    dt = time.time()-time0
                    print(epoch, '/' , args.num_epochs, " ",
                          m,'/', current_nlos_data.shape[1], " ",
                          n, '/', current_nlos_data.shape[2], " ",
                          'histogram loss:', loss_batch.item(), " ",
                          'time:', dt)
                    time0 = time.time()
                    
            
                
        # save models at the end of the training 
        if epoch == args.num_epochs - 1:
            save_model(out_dir, sdf_network, epoch+1, 0, args, model_name="sdf", print_path=True)
            save_model(out_dir, color_network, epoch+1, 0, args, model_name="color", print_path=True)
            save_model(out_dir, deviation_network, epoch+1, 0, args, model_name="deviation", print_path=True)
            if args.render_background == 1:
                save_model(out_dir, background_network, epoch+1, 0, args, model_name="background", print_path=True)
         

            
                            
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
