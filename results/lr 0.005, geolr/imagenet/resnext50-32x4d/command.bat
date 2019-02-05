python imagenet.py -a resnext50 --base-width
4 --cardinality 32 --data /dd/data/ --epochs 120 --schedule 31 61 --gamma 0.1 -c checkpoint s/imagenet/resnext50-32x4d --lr 0.005 --train-batch 64