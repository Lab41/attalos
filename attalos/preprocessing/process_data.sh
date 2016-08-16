#!/bin/bash

basepath=$1
dataset=$2
datetime=`date +%Y%m%d`

export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../../"

echo $PYTHONPATH

if [ $# -ne 2 ]
then
   echo "Script to help with generating features for follow on processing"
   echo "Current supported datasets: mscoco, visualgenome, iaprtc, espgame"
   echo -e "\nUsage: \n$0 /path/to/where/data/goes dataset\n"
   exit 1
fi


if [ ! -d "$basepath/datasets" ]
then
    mkdir "$basepath/datasets"
fi

if [ ! -d "$basepath/datasets/$dataset" ]
then
    mkdir "$basepath/datasets/$dataset"
fi

if [ ! -d "$basepath/features" ]
then
    mkdir "$basepath/features"
    mkdir "$basepath/features/image"
    mkdir "$basepath/features/text"
fi

echo "Extracting training text features"
python2 attalos/preprocessing/text/extract_text_features.py \
       --dataset_dir "$basepath/datasets/$dataset" \
       --output_fname "$basepath/features/text/"$dataset"_train_"$datetime"_text.json.gz" \
       --dataset_type "$dataset" \
       --split "train"

echo "Extracting test text features"
python2 attalos/preprocessing/text/extract_text_features.py \
       --dataset_dir "$basepath/datasets/$dataset" \
       --output_fname "$basepath/features/text/"$dataset"_test_"$datetime"_text.json.gz" \
       --dataset_type "$dataset" \
       --split "test"

echo "Extracting training inception image features"
python2 attalos/preprocessing/image/extract_inception_features.py \
       --dataset_dir "$basepath/datasets/$dataset" \
       --output_fname "$basepath/features/image/"$dataset"_train_"$datetime"_inception.hdf5" \
       --dataset_type "$dataset" \
       --split "train"

echo "Extracting test inception image features"
python2 attalos/preprocessing/image/extract_inception_features.py \
       --dataset_dir "$basepath/datasets/$dataset" \
       --output_fname "$basepath/features/image/"$dataset"_test_"$datetime"_inception.hdf5" \
       --dataset_type "$dataset" \
       --split "test"

if [ -e "$PYTHONPATH/vgg16-20160129.tfmodel" ]
then
    echo "Extracting training vgg image features"
    python2 attalos/preprocessing/image/extract_vgg_features.py \
       --dataset_dir "$basepath/datasets/$dataset" \
       --output_fname "$basepath/features/image/"$dataset"_train_"$datetime"_vgg.hdf5" \
       --dataset_type "$dataset" \
       --split "train"

    echo "Extracting test vgg image features"
    python2 attalos/preprocessing/image/extract_vgg_features.py \
       --dataset_dir "$basepath/datasets/$dataset" \
       --output_fname "$basepath/features/image/"$dataset"_test_"$datetime"_vgg.hdf5" \
       --dataset_type "$dataset" \
       --split "test"
else
    echo "VGG model file not found in $PYTHONPATH/vgg16-20160129.tfmodel please download or create per instructions"
    echo "Instructions: https://github.com/ry/tensorflow-vgg16"
fi
