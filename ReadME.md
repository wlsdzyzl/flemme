# Flemme: A **FLE**xible and **M**odular Learning Platform for **ME**dical Images
![](./images/flemme.gif)

## Overview
Flemme is a flexible and modular learning platform for medical images. In Flemme, we separate encoders from the model architectures, enabling fast model construction via different combinations for medical image segmentation, reconstruction and generation. In addition, a general hierarchical architecture with a pyramid loss is proposed for vertical feature refinement and integration. Please see [documentation of Flemme](https://flemme-docs.readthedocs.io/en/latest/) for more information.


![](./images/overview.png)

We are also working on Flemme to support *point cloud* modeling. An illustration of models built with `PointMamba2Encoder` is presented in the following figure:

![](./images/pointmamba2.png)

### Supported architectures

![](./images/archis.png)

### Hierarchical architecture with a pyramid Loss (H-SeM + UNet)
<p align="center">
<img src=./images/pyramid.png width=80%/>
</p>

## Get started with Flemme
### Requirement list
#### Basic:
```
torch torchvision simpleitk nibabel matplotlib scikit-image scikit-learn tensorboard tqdm
```
#### For vision transformer:
```
einops
```
#### For vision mamba: 
```
mamba-ssm (CUDA version >= 11.6)
```
If you have trouble to install `mamba-ssm`, you can download the corresponding `.whl` from https://github.com/state-spaces/mamba/releases based on your cuda and torch version. We recommend you to use the version with `ABI=FALSE`. Then run the following command:
```
python install mamba-ssmxxxxx.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```
#### For point cloud:
```
POT plyfile KNN-CUDA fpsample geomloss trimesh
```
KNN-CUDA is from https://github.com/unlimblue/KNN_CUDA.
#### For graph:
```
torch_geometric torch-cluster
```
You can modify `flemme/config.py` to disable some components of Flemme so that you don't need to install the corresponding required packages.

### Setup

Git clone from git@github.com:wlsdzyzl/flemme.git

Run following commands in terminal to setup Flemme to your environment:
```
cd flemme
python setup.py install
```
### Usage
Creating a deep learning model with Flemme is quite straightforward; you don't need to write any code. All things can be down through a `yaml` config file. An example of constructing a segmentation model with UNet encoder using ResConvBlock looks like:
```yaml
model:
  ### architecture
  name: SeM
  ### encoder
  encoder:
    name: UNet
    image_size: [320, 256]
    in_channel: 3
    out_channel: 1
    patch_channel: 32
    patch_size: 2
    down_channels: [64, 128, 256]
    middle_channels: [512, 512]
    building_block: res_conv
    normalization: batch
  ### loss function
  segmentation_losses: 
    - name: Dice
    - name: BCEL

```
You may also need to specify the data-loader, optimizer, checkpoint path and other hyper-parameters. A full configuration refers to `resources/img/biomed_2d/cvccdb/train_unet_sem.yaml`. To train the model, run command:
```
train_flemme --config path/to/train_config.yaml
```
For visualization of the training process:
```
tensorboard --logdir path/to/ckp/
```
For testing:
```
test_flemme --config path/to/test_config.yaml
```
**Supported encoders**: 
- [CNN, UNet, ViT, ViTU, Swin, SwinU, VMamba, VMambaU] for 2D/3D image 
- [PointNet, PointTrans, PointMamba, PointNet2, PointTrans2, PointMamba2] for point cloud.

*A encoder named as XXU indicates it's a U-shaped encoder.*

*UNet is an alias of CNNU.*

**Supported Architectures**: 
- [ClM, ] for classification,
- [SeM, HSeM] for segmentation, 
- [AE, HAE, SDM] for reconstruction, 
- [VAE, DDPM, DDIM, EDM, LDM] for generation. 

*DDIM refers to denoising diffusion implicit model, which is a fast sample strategy.*

*EDM refers to elucidated diffusion models from paper "Elucidating the Design Space of Diffusion-Based Generative Models"*

*SDM refers to supervised diffusion model (use input as a input condition of ddpm).*

*LDM refers to latent diffusion model, constructed with a auto-encoder/vae and ddpm.*

A detailed instruction of supported encoders, context embeddings, model architectures and training process can refer to [documentation of flemme](https://flemme-docs.readthedocs.io/en/latest/).

## Results

### 2D/3D Image

For segmentation, we evaluate our methods on six public datasets: **CVC-ClinicDB, Echonet, ISIC, TN3K, BraTS21 (3D), ImageCAS (3D)**.

For reconstruction, we evaluate our methods on **FastMRI**.

Configuration files are in `resources/img/biomed_2d` and `resources/img/biomed_3d`.

#### Segmentation results

![](./images/seg_res.png)

#### Reconstruction & Generation results

![](./images/recon_res.png)

### Point Cloud

#### Completion & Segmentation results

![](./images/pcd_result.png)

To train and evaluate the model proposed by [2], run the following commands (You can change flemme to an old commit `c978a59`):

```bash
## classification
train_flemme --config /path/to/project/flemme/resources/pcd/medpoints/cls/train_pointmamba2knn_clm.yaml
test_flemme --config /path/to/project/flemme/resources/pcd/medpoints/cls/test_pointmamba2knn_clm.yaml
## completion
train_flemme --config /path/to/project/flemme/resources/pcd/medpoints/cpl/train_pointmamba2knn_cpl.yaml
test_flemme --config /path/to/project/flemme/resources/pcd/medpoints/cpl/test_pointmamba2knn_cpl.yaml
## segmentation
train_flemme --config /path/to/project/flemme/resources/pcd/medpoints/seg/train_pointmamba2knn_sem.yaml
test_flemme --config /path/to/project/flemme/resources/pcd/medpoints/seg/test_pointmamba2knn_sem.yaml
```

## *MedPointS* Dataset

MedPointS is a large-scale medical point cloud dataset based on MedShapeNet for anatomy classification, completion, and segmentation.

An overview of MedPointS is presented int the following figure:

![](./images/medpoints.png)

You can download *MedPointS* from this [link](https://pan.baidu.com/s/1OKiglb6FtGmBLNwhVQXz9Q?pwd=cs27). 

Alternatively, you can use load the dataset from Hugging Face: [MedPoints-cls](https://huggingface.co/datasets/wlsdzyzl/MedPointS-cls) , [MedPoints-cpl](https://huggingface.co/datasets/wlsdzyzl/MedPointS-cpl), and [MedPoints-seg](https://huggingface.co/datasets/wlsdzyzl/MedPointS-seg) for classification, completion, and segmentation tasks. 

## Play with Flemme
#### Toy Example for Diffusion model
Configuration file: `resources/toy_ddpm.yaml`
```
train_flemme --config resources/toy_ddpm.yaml
```
![](./images/ddpm_toy.png)
#### MINST 
Configuration files are in  `resources/img/mnist`
**AutoEncoder** & **Variational AutoEncoder**

![](./images/ae_mnist.png)

**Denoising Diffusion Probabilistic Model**
![](./images/ddpm_mnist.png)
#### CIFA10 
Configuration files are in `resources/img/cifar10`
**AutoEncoder** & **Variational AutoEncoder**

![](./images/ae_cifar10.png)

**Denoising Diffusion Probabilistic Model (conditional)**
![](./images/cddpm_cifar10.png)
## BibTeX
If you find our project helpful, please consider to cite the following works:

[1] Flemme: A Flexible and Modular Learning Platform for Medical Images; BIBM 2024.
```
@misc{zhang2024flemmeflexiblemodularlearning,
      title={Flemme: A Flexible and Modular Learning Platform for Medical Images}, 
      author={Guoqing Zhang and Jingyun Yang and Yang Li},
      year={2024},
      eprint={2408.09369},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2408.09369}, 
}
```
[2] Hierarchical Feature Learning for Medical Point Clouds via State Space Model; MICCAI 2025.
```
@misc{zhang2025hierarchicalfeaturelearningmedical,
      title={Hierarchical Feature Learning for Medical Point Clouds via State Space Model}, 
      author={Guoqing Zhang and Jingyun Yang and Yang Li},
      year={2025},
      eprint={2504.13015},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.13015}, 
}
```
## Acknowledgement
Thanks to [mamba](https://github.com/state-spaces/mamba), [swin-transformer](https://github.com/microsoft/Swin-Transformer), [diffusion model](https://github.com/lucidrains/denoising-diffusion-pytorch), and
[pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch) for their wonderful works.
