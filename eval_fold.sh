#!/bin/bash
# Evaluate Fold / AutoFold variants on the OpenOOD benchmarks.
#
# Usage:
#   bash eval_fold.sh                      # all datasets x all methods
#   bash eval_fold.sh cifar10              # all methods on CIFAR-10
#   bash eval_fold.sh cifar10 fold         # single run
#   ARCH=regnet bash eval_fold.sh imagenet fold   # other ImageNet-1K backbone
#
# Datasets: cifar10 | cifar100 | imagenet200 | imagenet
# Methods : fold | fold_react | fold_ash | autofold | autofold_react | autofold_ash
# ARCH    : ImageNet-1K backbone (default resnet50)
#           resnet50 | regnet | densenet-121 | wrn50 | resnext50
#
# Checkpoints are expected under ./results (see README).

BATCHSIZE=${BATCHSIZE:-256}
ARCH=${ARCH:-resnet50}

DATASETS=(cifar10 cifar100 imagenet200 imagenet)
METHODS=(fold fold_react fold_ash autofold autofold_react autofold_ash)

if [ -n "$1" ]; then DATASETS=("$1"); fi
if [ -n "$2" ]; then METHODS=("$2"); fi

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        echo "============================================"
        echo "${DATASET} / ${METHOD}"
        echo "============================================"

        if [ "$DATASET" == 'imagenet' ]; then
            python scripts/eval_ood_imagenet.py \
                --batch-size ${BATCHSIZE} \
                --tvs-pretrained \
                --arch ${ARCH} \
                --postprocessor ${METHOD} \
                --save-csv
        else
            case $DATASET in
                cifar10)     ROOT=./results/cifar10_resnet18_32x32_base_e100_lr0.1_default ;;
                cifar100)    ROOT=./results/cifar100_resnet18_32x32_base_e100_lr0.1_default ;;
                imagenet200) ROOT=./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default ;;
                *) echo "Unsupported dataset: $DATASET"; continue ;;
            esac
            python scripts/eval_ood.py \
                --batch-size ${BATCHSIZE} \
                --id-data ${DATASET} \
                --root ${ROOT} \
                --postprocessor ${METHOD} \
                --save-csv
        fi
    done
done
