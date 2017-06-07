# Training recipes

## CIFAR-10

ResNet-110
```sh
CUDA_VISIBLE_DEVICES=0,1 python cifar.py -a  -dataset cifar10 -nGPU 2 -batchSize 128 -depth 110
```