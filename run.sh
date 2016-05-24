#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: run.sh <work-directory> <command>"
  exit
fi

docker run -it --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia0 -v $1:/work/ l41-caffe-keras-tensorflow $2
