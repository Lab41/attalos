#!/bin/bash

docker run -it\
   --device /dev/nvidiactl:/dev/nvidiactl \
   --device /dev/nvidia-uvm:/dev/nvidia-uvm \
   --device /dev/nvidia0:/dev/nvidia0 \
   --volume ~/:/work \
   --volume /tmp/tensorboard:/tmp/tensorboard \
   -p 1088:8888 \
   -p 2888:6006 \
   l41-tensorflow /bin/bash

# Inside the docker container, run these commands:
# > jupyter notebook --ip='*' &
# > tensorboard --logdir=/tmp/tensorboard

