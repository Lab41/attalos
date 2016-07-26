#!/bin/bash

docker run -d -p 9999:8888 -v ~/:/work --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
                   --name karls-notebooks --device /dev/nvidia1:/dev/nvidia0 -it l41-fullenv /bootstrap.sh
