GPUID=$1
DATASET=$2
NET=$3

cd ..

CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}20 --checkpoint checkpoint/$DATASET/${NET}20 --manualSeed 1234
CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}32 --checkpoint checkpoint/$DATASET/${NET}32 --manualSeed 1234
CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}44 --checkpoint checkpoint/$DATASET/${NET}44 --manualSeed 1234
CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}56 --checkpoint checkpoint/$DATASET/${NET}56 --manualSeed 1234
CUDA_VISIBLE_DEVICES=$GPUID python cifar.py -d $DATASET -a ${NET}110 --checkpoint checkpoint/$DATASET/${NET}110 --manualSeed 1234

cd -