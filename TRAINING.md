
## CIFAR-10

#### AlexNet
```
python cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/alexnet 
```


#### VGG19 (BN)
```
python cifar.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn 
```

#### Resnet-110
```
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 
```

#### Resnet-1202
```
python cifar.py -a resnet --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-1202 
```

#### ResNeXt-29, 8x64d
```
python cifar.py -a resnext --depth 29 --cardinality 8 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-8x64d 
```
#### ResNeXt-29, 16x64d
```
python cifar.py -a resnext --depth 29 --cardinality 16 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-16x64d 
```

#### WRN-28-10-drop
```
python cifar.py -a wrn --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoints/cifar10/WRN-28-10-drop
```

## CIFAR-100

#### AlexNet
```
python cifar.py -a alexnet --dataset cifar100 --checkpoint checkpoints/cifar100/alexnet --epochs 164 --schedule 81 122 --gamma 0.1 
```

#### VGG19 (BN)
```
python cifar.py -a vgg19_bn --dataset cifar100 --checkpoint checkpoints/cifar100/vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 
```

#### Resnet-110
```
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 
```

#### Resnet-1202
```
python cifar.py -a resnet --dataset cifar100 --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-1202 
```

#### ResNeXt-29, 8x64d
```
python cifar.py -a resnext --dataset cifar100 --depth 29 --cardinality 8 --widen-factor 4 --checkpoint checkpoints/cifar100/resnext-8x64d --schedule 150 225 --wd 5e-4 --gamma 0.1
```
#### ResNeXt-29, 16x64d
```
python cifar.py -a resnext --dataset cifar100 --depth 29 --cardinality 16 --widen-factor 4 --checkpoint checkpoints/cifar100/resnext-16x64d --schedule 150 225 --wd 5e-4 --gamma 0.1
```

#### WRN-28-10-drop
```
python cifar.py -a wrn --dataset cifar100 --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoints/cifar100/WRN-28-10-drop
```


## ImageNet
### ResNet-101
```
python imagenet.py -a resnet101 --data ~/dataset/ILSVRC2012/ ---epochs 90 --schedule 31 61 --gamma 0.1 -j 12 -c checkpoints/imagenet/resnet101
```