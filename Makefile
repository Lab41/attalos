build: depends
	docker build -t l41-nvidia-base -f Dockerfile.nvidia .
	docker build -t l41-theano -f Dockerfile.theano .
	docker build -t l41-keras -f Dockerfile.keras .
	docker build -t l41-tensorflow -f Dockerfile.tf .
	docker build -t l41-caffe -f Dockerfile.caffe .
	docker build -t l41-domino-tensorflow -f Dockerfile.domino .
	docker build -t l41-torch -f Dockerfile.torch .
	docker build -t l41-densecap -f Dockerfile.densecap .


attalos-bash: depends
	docker run --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        	   --device /dev/nvidia0:/dev/nvidia0  -it l41-tensorflow /bin/bash

attalos-torch: depends
	docker run --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
                   --device /dev/nvidia0:/dev/nvidia0  -it l41-torch /bin/bash

attalos-theano: depends
	docker run --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
                   --device /dev/nvidia2:/dev/nvidia0  -it l41-keras /bin/bash
attalos-densecap-bash: depends
	docker run --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
                   --device /dev/nvidia0:/dev/nvidia0  -it l41-densecap /bin/bash

notebook: depends
	# docker build -t l41-attalos-notebook -f Dockerfile.notebook .
	docker build -t l41-attalos-notebook-conda -f Dockerfile.notebook .
	docker run -d -p 9999:8888 -v /data:/data -v ~/:/work --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        	   --device /dev/nvidia0:/dev/nvidia0 -it l41-attalos-notebook /bootstrap.sh

depends:
	@echo
	@echo "checking dependencies"
	@echo
	docker -v
