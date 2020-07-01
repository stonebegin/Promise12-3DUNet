# pytorch-3dunet

PyTorch implementation of a standard 3D U-Net based on:

[3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) 
Özgün Çiçek et al.

as well as Residual 3D U-Net based on:

[Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120.pdf) Kisuk Lee et al.

## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

## Run
python train.py --config ./resources/train_config_4d_input.yaml
