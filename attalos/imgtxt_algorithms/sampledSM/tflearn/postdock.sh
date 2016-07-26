#!/bin/bash

curdir=$PWD

# Inside the docker container, run these commands:
cd /work
jupyter notebook --ip='*' &
tensorboard --logdir=/tmp/tensorboard &

cd $curdir
