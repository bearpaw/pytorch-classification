# pytorch-classification
Classification on with PyTorch.

## CIFAR-10 / CIFAR-100
Since the size of images in CIFAR dataset is `32x32`, popular network structures for ImageNet need some modifications to adapt this input size. The modified models is located in the subfolder `models`:
- [x] [AlexNet](https://arxiv.org/abs/1404.5997)
- [x] [VGG](https://arxiv.org/abs/1409.1556)
- [x] [ResNet](https://arxiv.org/abs/1512.03385)
- [x] [Pre-activated ResNet](https://arxiv.org/abs/1603.05027)
- [x] [Hourglass Network](https://arxiv.org/abs/1603.06937)
- [x] [Residual Attention Networks](https://arxiv.org/abs/1704.06904)
- [ ] [Inception (v3)](http://arxiv.org/abs/1512.00567)
- [ ] [DenseNet](https://arxiv.org/abs/1608.06993)
- [ ] [SqueezeNet](https://arxiv.org/abs/1602.07360)
- [ ] [ResNeXt](https://arxiv.org/abs/1611.05431)

### Results

Fixed random seed: 1234

| Model              | CIFAR-10 (%)       | CIFAR-100 (%)      |
| ------------------ | ------------------ | ------------------ |
| alexnet            | 75.81              | 41.23              |
| vgg16_bn           | 92.93              | 72.62              |
| vgg19_bn           | 92.82              | 71.31              |
| ------------------ | ------------------ | ------------------ |
| resnet20           | 91.62              | xx.xx              |
| resnet32           | 92.70              | xx.xx              |
| resnet44           | 93.27              | xx.xx              |
| resnet56           |               | xx.xx              |
| resnet110          |               | xx.xx              |
| ------------------ | ------------------ | ------------------ |
| preresnet20        | 91.67              | xx.xx              |
| preresnet32        | 92.68              | xx.xx              |
| preresnet44        |               | xx.xx              |
| preresnet56        |               | xx.xx              |
| preresnet110       |               | xx.xx              |