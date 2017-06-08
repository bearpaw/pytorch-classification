# pytorch-classification
Classification on CIFAR10/100 with PyTorch.

## Install
* Install [PyTorch](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/bearpaw/pytorch-classification.git
  ```
* Run experiments by following the [Training recipes](#training-recipes)

## Results

Top1 error.

| Model                | Params (M, CIFAR10)|  CIFAR-10 (%)      | CIFAR-100 (%)      |
| -------------------  | ------------------ | ------------------ | ------------------ |
| alexnet              | 2.47               | 22.78              | 56.13              |
| vgg19_bn             | 20.04              | 6.66               | 28.05              |
| Resnet-110           | 1.70               | local               |     local          |
| Resnet-1202          | 18.58              | 186               |       186        |
| ResNeXt-29, 8x64     | 34.43              | 3.62               |               |
| ResNeXt-29, 16x64    | 68.16              | 164              |             10137  |
| WRN-28-10 (drop 0.3) | 36.48              | 179              |             10137  |

## Datasets

### CIFAR-10 / CIFAR-100
Since the size of images in CIFAR dataset is `32x32`, popular network structures for ImageNet need some modifications to adapt this input size. The modified models is located in the subfolder `models`:
- [x] [AlexNet](https://arxiv.org/abs/1404.5997)
- [x] [VGG](https://arxiv.org/abs/1409.1556) (Imported from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar))
- [ ] [ResNet](https://arxiv.org/abs/1512.03385)
- [x] [ResNeXt](https://arxiv.org/abs/1611.05431) (Imported from [ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch))
- [x] [Wide Residual Networks](http://arxiv.org/abs/1605.07146) (Imported from [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch))
- [ ] [DenseNet](https://arxiv.org/abs/1608.06993)

## Training recipes
Please see the [Training recipes](TRAINING.md) for how to train the models.

```
## TODO
- [ ] Add ImageNet