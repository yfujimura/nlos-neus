# NLOS-NeuS: Non-line-of-sight Neural Implicit Surface
This is the official pytorch implementation of "**NLOS-NeuS: Non-line-of-sight Neural Implicit Surface**," ICCV2023.

<!--<h3 align="center"> -->

### [Project Page](https://yfujimura.github.io/nlos-neus/) | [Paper](https://arxiv.org/pdf/2303.12280.pdf) | [Poster](./docs/static/pdfs/poster.pdf)

## Requirements
Our experimental environment is
```
Ubuntu 20.04
CUDA 11.3.1
pytorch 1.11.0
```
The requirements can be installed by
```
conda install --name nlos-neus --file requirements.yml
conda activate nlos-neus
```

## Data
Please download the preprocessed data from the [NeTF](https://github.com/zeromakerplus/NeTF_public).

## Pre-trained models
We provide our pre-trained model [here](https://drive.google.com/drive/folders/11W-XyuyK0X9gdQ6dpRmYDV7M9MlKGAmJ?usp=sharing).
```
tar -zxvf out.tar.gz
```

## Test
We provide several codes for obtaining results.
```
# render depth saved as npy
python render_depth.py --config configs/test/zaragoza_bunny.txt --test_volume_size 207
# render directional albedo saved as npy
python render_albedo.py --config configs/test/zaragoza_bunny.txt
# extract point cloud and mesh saved as ply
python extract_mesh.py --config configs/test/zaragoza_bunny.txt
```
These results are saved at ```recon```.

## Training
If you want to run training, please run the following for estimating SDF lower bounds beforehand:
```
python space_carving --scene zaragoza_bunny
```
Then, please run
```
python run_netf.py --config configs/zaragoza_bunny.txt
```

## Acknowledgments
This code was heavily built on [NeTF_public](https://github.com/zeromakerplus/NeTF_public). We are grateful for their excellent work.