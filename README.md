# pytorch-classification
Classification on CIFAR10/100 with PyTorch.

## Install
* Install [PyTorch](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/bearpaw/pytorch-classification.git
  ```

## Results

### CIFAR
Top1 error rate on CIFAR10/100 are reported. You may get different results when training your models with different random seed.

| Model                | Params (M, CIFAR10)|  CIFAR-10 (%)      | CIFAR-100 (%)      |
| -------------------  | ------------------ | ------------------ | ------------------ |
| alexnet              | 2.47               | 22.78              | 56.13              |
| vgg19_bn             | 20.04              | 6.66               | 28.05              |
| Resnet-110           | 1.70               | 6.11               | 28.86              |
| WRN-28-10 (drop 0.3) | 36.48              | 3.79               | 18.14              |
| ResNeXt-29, 8x64     | 34.43              | 3.69               | 17.38              |
| ResNeXt-29, 16x64    | 68.16              | 3.53               |             10137  |

### ImageNet
Single-crop (224x224) validation error rate


| Model                | Params (M)         |  Top-1 Error (%)   | Top-5 Error  (%)   |
| -------------------  | ------------------ | ------------------ | ------------------ |
| Resnet-101           | 44.55              |                    |                    |


## Supported Architectures

### CIFAR-10 / CIFAR-100
Since the size of images in CIFAR dataset is `32x32`, popular network structures for ImageNet need some modifications to adapt this input size. The modified models is located in the subfolder `models`:
- [x] [AlexNet](https://arxiv.org/abs/1404.5997)
- [x] [VGG](https://arxiv.org/abs/1409.1556) (Imported from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar))
- [x] [ResNet](https://arxiv.org/abs/1512.03385)
- [x] [ResNeXt](https://arxiv.org/abs/1611.05431) (Imported from [ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch))
- [x] [Wide Residual Networks](http://arxiv.org/abs/1605.07146) (Imported from [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch))
- [ ] [DenseNet](https://arxiv.org/abs/1608.06993)

### ImageNet
- [x] All models in `torchvision.models` (alexnet, vgg, resnet, densenet, inception_v3, squeezenet)
- [ ] [ResNeXt](https://arxiv.org/abs/1611.05431)
- [ ] [Wide Residual Networks](http://arxiv.org/abs/1605.07146)

## Training recipes
Please see the [Training recipes](TRAINING.md) for how to train the models.