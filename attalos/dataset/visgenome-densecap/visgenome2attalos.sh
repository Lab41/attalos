#!/bin/bash

PYTHONPATH=$PYTHONPATH:'../../../'

# wget https://raw.githubusercontent.com/Lab41/attalos/master/attalos/imgtxt_algorithms/densecap/info/densecap_splits.json
ln -s ../../../attalos/imgtxt_algorithms/densecap/info/densecap_splits.json densecap_splits.json
python visgenome2attalos.py

mkdir notfounds
mv vis*notfound notfounds/

touch visgenome.txt
cat visgenome_training.txt >> visgenome.txt
# cat visgenome_validation.txt >> visgenome.txt
cat visgenome_testing.txt >> visgenome.txt

awk '{print "/data/fs4/datasets/vg_unzipped/" $0}' visgenome.txt > vgfullpath.txt

python ../../preprocessing/text/extract_text_features.py \
    --dataset_dir vgfullpath.txt --output_fname visgenome-densecap-train.json.gz \
    --dataset_type generic --split train      

python ../../preprocessing/text/extract_text_features.py \
    --dataset_dir vgfullpath.txt --output_fname visgenome-densecap-test.json.gz \
    --dataset_type generic --split test

# Suppress deprecation warning since it occurs once per image per batch normalization in the inception graph (aka a lot)
python ../../preprocessing/image/extract_inception_features.py \
    --dataset_dir vgfullpath.txt \
    --output_fname visualgenome-densecap-train.hdf5 \
    --dataset_type generic \
    --split train

python ../../preprocessing/image/extract_inception_features.py \
    --dataset_dir vgfullpath.txt \
    --output_fname visualgenome-densecap-test.hdf5 \
    --dataset_type generic \
    --split test

rm visgenome_training.txt visgenome_validation.txt visgenome_testing.txt
rm -rf notfounds
