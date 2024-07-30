# Image segmentation using text prompts

This repository contains the code used in the paper "Image Segmentation Using Text and Image Prompts".

![image](https://github.com/user-attachments/assets/274a2a1f-b0c8-4057-b52c-2e115fdbfbd0)

The systems allows to create segmentation models without training based on:

+ An arbitrary text query

**Quick Start**

In the Quickstart.ipynb notebook we provide the code for using a pre-trained CLIPSeg model. If you run the notebook locally, make sure you downloaded the rd64-uni.pth weights, either manually or via git lfs extension. It can also be used interactively using MyBinder (please note that the VM does not use a GPU, thus inference takes a few seconds).

**Dependencies**

This code base depends on pytorch, torchvision and clip (pip install git+https://github.com/openai/CLIP.git). Additional dependencies are hidden for double blind review.

**Models**

CLIPDensePredT: CLIPSeg model with transformer-based decoder.

ViTDensePredT: CLIPSeg model with transformer-based decoder.

**Finetune**

We finetune original CLIPSeg with new loss function by adding Dice Loss to BCE Loss with logits to enhance edge detection availability.

Streamlit demo can be use to visualize the finetune result.
