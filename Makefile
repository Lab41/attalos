build: depends
	docker build -t l41-nvidia-base -f Dockerfile.nvidia .
	docker build -t l41-theano -f Dockerfile.theano .
	docker build -t l41-keras -f Dockerfile.keras .
	docker build -t l41-tensorflow -f Dockerfile.tf .
	docker build -t l41-caffe -f Dockerfile.caffe .
	docker build -t l41-domino-tensorflow -f Dockerfile.domino .

attalos-bash: depends
	docker run --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        	   --device /dev/nvidia0:/dev/nvidia0  -it l41-tensorflow /bin/bash

notebook: depends
	docker build -t l41-attalos-notebook -f Dockerfile.notebook .
	docker run -d -p 8888:8888 -v /data:/data -v ~/:/work --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        	   --device /dev/nvidia0:/dev/nvidia0 -it l41-attalos-notebook /bootstrap.sh

depends:
	@echo
	@echo "checking dependencies"
	@echo
	docker -v
