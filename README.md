[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# milliMap
### [Project](https://christopherlu.github.io/publications/millimap) | [Youtube](https://www.youtube.com/watch?v=VnxS-jsr4vk&feature=youtu.be) | [Paper](https://christopherlu.github.io/files/papers/[MobiSys2020]milliMap.pdf) <br>
Pytorch implementation of our method for 2D indoor dense mapping via a low-cost, off-the-shelf mmWave radar ([TI AWR1443](https://www.ti.com/product/AWR1443)). Our method can reconstruct a dense grid map with accuracy comparable to a lidar. <br><br>
[See Through Smoke: Robust Indoor Mapping with Low-cost mmWave Radar](https://christopherlu.github.io/publications/millimap)  
Chris Xiaoxuan Lu, Stefano Rosa, Peijun Zhao, Bing Wang, Changhao Chen, John A. Stankovic, Niki Trigoni and Andrew Markham
In [MobiSys 2020](https://www.sigmobile.org/mobisys/2020/).  

## Prerequisites
- Linux or macOS
- Python 2.7 or 3.6
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate).
```bash
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/ChristopherLu/milliMap
cd milliMap
```

### Dataset
- To train and test a model on the full dataset, please download it from the [here](https://www.dropbox.com/s/ap54f319vpttaat/dataset.zip?dl=0) (dropbox link).
After downloading and unzip, please put the dataset folder in this project directory.

### Testing
- Please download the pre-trained milliMap model from [here](https://www.dropbox.com/s/nfwq1cjcnznadzc/best_net_G.pth?dl=0) (dropbox link), and put it under `./checkpoints/full_line_10/`
- For example, the following scripts tests the model on building A (`bash ./scripts/test_cross_floor.sh`):
```#!/bin/bash
python test.py --name full_line_10 --dataroot ./dataset/test/cross_floor --label_nc 0 --gpu_ids 1 --batchSize 4 --loadSize 256 --fineSize 256 --no_instance --which_epoch best
```
The test results will be summarized to a html file here: `./results/full_line_10/test_best/index.html`, with generated images in the same level of folder.

More example scripts can be found in the `scripts` directory.

### Training
- Train a model (`bash ./scripts/train_millimap.sh`) in which a line detector is used and the hyper-parameter of prior loss is set to 10:
```############## To train the model with new line detectors #############
python train.py --name full_line_10 \
    --dataroot ./dataset/train/ \
    --label_nc 0 \
    --gpu_ids 0 \
    --batchSize 16 \
    --loadSize 256 \
    --fineSize 256 \
    --no_instance \
    --lambda_prior 10 \
    --detector_type line \
    --tf_log
```
- To view training results, please checkout intermediate results in `./checkpoints/full_line_10/web/index.html`.
- If you have tensorflow installed, you can see tensorboard logs in `./checkpoints/full_line_10/logs` by adding `--tf_log` to the training scripts.
- If you have multiple GPUs, you can use them to speed up training by `--gpu_ids 0,1,2,3,4,5,6,7` to the training scripts. Note: this is not tested and we trained our model using single GPU only. Please use at your own discretion.
- You can play around different training configuration by referring to the [training options](https://github.com/ChristopherLu/milliMap/blob/master/options/train_options.py) file.  

### Training with your own dataset
When training on customized dataset is needed, please change the dataroot filed in the training scripts accordingly. Please go to [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) first to see how to prepare a dataset in valid format.

## More Training/Test Details
- Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.

## Citation

If you find this useful for your research, please use the following.

```
@inproceedings{lu2020millimap,
  title={See Through Smoke: Robust Indoor Mapping with Low-cost mmWave Radar},
  author={Chris Xiaoxuan Lu, Stefano Rosa, Peijun Zhao, Bing Wang, Changhao Chen, John A. Stankovic, Niki Trigoni and Andrew Markham},  
  booktitle={ACM International Conference on Mobile Systems, Applications, and Services (MobiSys)},
  year={2020}
}
```

## Acknowledgments
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
