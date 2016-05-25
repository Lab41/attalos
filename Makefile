build: depends
	docker build -t l41-nvidia-base -f Dockerfile.nvidia .
	docker build -t l41-theano-base -f Dockerfile.theano .
	docker build -t l41-keras-base -f Dockerfile.keras .
	docker build -t l41-caffe-keras-tf -f Dockerfile.caffe-keras-tf .

attalos-bash: depends
	docker run -it l41-caffe-keras-tf /bin/bash

depends:
	@echo
	@echo "checking dependencies"
	@echo
	docker -v
