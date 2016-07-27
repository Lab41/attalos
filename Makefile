all: tensorflow theano caffe torch

# Base requirements for all containers
depends:
	@echo
	@echo "checking dependencies"
	@echo
	docker -v

nvidia-base: depends
	docker build -t l41-nvidia-base -f Dockerfile.nvidia .

# Specific deep learning frameworks
tensorflow: nvidia-base
	docker build -t l41-tensorflow -f Dockerfile.tf .
	docker build -t l41-domino-tensorflow -f Dockerfile.domino .

theano: nvidia-base
	docker build -t l41-theano -f Dockerfile.theano .
	docker build -t l41-keras -f Dockerfile.keras .

caffe: nvidia-base
	docker build -t l41-caffe -f Dockerfile.caffe .

torch: nvidia-base
	docker build -t l41-torch -f Dockerfile.torch .

# Run various environments
attalos-bash: tensorflow
	docker run --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        	   --device /dev/nvidia0:/dev/nvidia0  -it l41-tensorflow /bin/bash

attalos-torch: torch
	docker run --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
                   --device /dev/nvidia0:/dev/nvidia0  -it l41-torch /bin/bash

notebook: tensorflow
	docker build -t l41-attalos-notebook -f Dockerfile.notebook .
	docker run -d -p 8888:8888 -v /data:/data -v ~/:/work --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        	   --device /dev/nvidia0:/dev/nvidia0 -it l41-attalos-notebook /bootstrap.sh
