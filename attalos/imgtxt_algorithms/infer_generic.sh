#!/bin/bash

basepath=$1
dataset=$2
model_fname=$3
w2v_fname=$4
fname=`echo ${dataset##*/}`
datetime=`date +%Y%m%d`


export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../../"
echo "Processing $dataset file"

if [ $# -ne 4 ]
then
   echo "Script to help with generating inferred features"
   echo -e "\nUsage: \n$0 /path/to/where/data/goes image_filelist model_fname w2v_fname\n"
   exit 1
fi


if [ ! -d "$basepath/features" ]
then
    mkdir "$basepath/features"
    mkdir "$basepath/features/image"
    mkdir "$basepath/features/text"
fi

echo "Extracting training text features"
python2 attalos/preprocessing/text/extract_text_features.py \
       --dataset_dir "$dataset" \
       --output_fname "$basepath/features/text/"$fname"_train_"$datetime"_text.json.gz" \
       --dataset_type "generic" \
       --split "train"

echo "Extracting training inception image features"
python2 attalos/preprocessing/image/extract_inception_features.py \
       --dataset_dir "$dataset" \
       --output_fname "$basepath/features/image/"$fname"_train_"$datetime"_inception.hdf5" \
       --dataset_type "generic" \
       --split "train"

echo "Infer data"
PYTHONPATH=$PWD python2 attalos/imgtxt_algorithms/infer.py    \
  "$basepath/features/image/"$fname"_train_"$datetime"_inception.hdf5"    \
  "$basepath/features/text/"$fname"_train_"$datetime"_text.json.gz" \
  "$basepath/$fname"_inference_"$datetime.hdf5" \
  "$w2v_fname" \
  --word_vector_type=glove --batch_size=1024 --model_type=negsampling  \
  --hidden_units=2048,1024 \
  --model_input_path="$model_fname"
