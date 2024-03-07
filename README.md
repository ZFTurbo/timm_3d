# PyTorch Volumes Models for 3D data 

Python library with Neural Networks for Volume (3D) Classification based on PyTorch.

This library is based on famous [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) library for images. Most of the documentation can be used directly from there. 

## Installation

* Type 1: `pip install timm_3d`
* Type 2: Copy `timm_3d` folder from this repository in your project folder.

## Quick start

You can create model as easy as:

```python
import timm_3d
import torch

m = timm_3d.create_model(
    'tf_efficientnet_b0.in1k',
    pretrained=True,
    num_classes=0,
    global_pool=''
)

# Shape of input (B, C, H, W, D). B - batch size, C - channels, H - height, W - width, D - depth
res = m(torch.randn(2, 3, 128, 128, 128))
print(f'Output shape: {res.shape}') 
```

* **Note 1**: you can use pretrained weights. They will be converted on the fly from 2D to 3D variant.
* **Note 2**: More examples can be found [here](test.py) 

## Models

Currently supported models in 3D variant

* Efficientnet family [[Code](timm_3d/models/efficientnet.py)]
* ResNet family [[Code](timm_3d/models/resnet.py)]
* CoAtNet and MaxVit family [[Code](timm_3d/models/maxxvit.py)]
* ConvNext family [[Code](timm_3d/models/convnext.py)]
* DenseNet family [[Code](timm_3d/models/densenet.py)]
* VGG family [[Code](timm_3d/models/vgg.py)]

[Full list of all possible models](docs/models_list.md)

## Notes for 3D version

### Input size

Recommended input size for backbones can be calculated as: `K = pow(N, 2/3)`. 
Where N - is size for input image for the same model in 2D variant.

For example for N = 224, K = 32. For N = 512, K = 64.

### Related repositories

 * [pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) - original 2D repo
 * [classification_models_3D](https://github.com/ZFTurbo/classification_models_3D) - 3D volumes classification models for keras/tensorflow
 * [segmentation_models_pytorch_3d](https://github.com/ZFTurbo/segmentation_models_pytorch_3d) - 3D volumes segmentation models for PyTorch
 * [volumentations](https://github.com/ZFTurbo/volumentations) - 3D augmentations

## Citation

If you find this code useful, please cite it as:
```
@article{solovyev20223d,
  title={3D convolutional neural networks for stalled brain capillary detection},
  author={Solovyev, Roman and Kalinin, Alexandr A and Gabruseva, Tatiana},
  journal={Computers in Biology and Medicine},
  volume={141},
  pages={105089},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2021.105089}
}
```

## To Do List
* Add support for more architectures
