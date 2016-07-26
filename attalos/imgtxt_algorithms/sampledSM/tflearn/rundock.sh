#!/bin/bash

docker run -it\
   --device /dev/nvidiactl:/dev/nvidiactl \
   --device /dev/nvidia-uvm:/dev/nvidia-uvm \
   --device /dev/nvidia0:/dev/nvidia0 \
   --volume /local_data:/local_data \
   --volume /home/ytesfaye/attalos:/attalos \
   --volume /tmp/tensorboard:/tmp/tensorboard \
   -p 8889:8888 \
   -p 6006:6006 \
   l41-tensorflow /bin/bash

# Inside the docker container, run these commands:
# > jupyter notebook --ip='*' &
# > tensorboard --logdir=/tmp/tensorboard

