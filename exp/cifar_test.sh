GPUID=$1
DATASET=$2
NET=$3

cd ..

CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}20 --resume checkpoint/$DATASET/${NET}20/model_best.pth.tar -e
CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}32 --resume checkpoint/$DATASET/${NET}32/model_best.pth.tar -e
CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}44 --resume checkpoint/$DATASET/${NET}44/model_best.pth.tar -e
CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}56 --resume checkpoint/$DATASET/${NET}56/model_best.pth.tar -e
CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}110 --resume checkpoint/$DATASET/${NET}110/model_best.pth.tar -e

cd -