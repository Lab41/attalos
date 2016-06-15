FROM l41-nvidia-base

# Install git, bc and dependencies
RUN apt-get update && apt-get install -y \
  git \
  bc \
  cmake \
  libatlas-base-dev \
  libatlas-dev \
  libboost-all-dev \
  libopencv-dev \
  libprotobuf-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  protobuf-compiler \
  libhdf5-dev \
  libleveldb-dev \
  liblmdb-dev \
  libsnappy-dev \
  gfortran > /dev/null

# Clone Caffe repo and move into it
RUN cd /root && git clone https://github.com/BVLC/caffe.git && cd caffe && \
# Install python dependencies
    cat python/requirements.txt | xargs -n1 pip install

RUN cd /root/caffe && \
# Make and move into build directory
  mkdir build && cd build && \
# CMake
  cmake .. && \
# Make
  make -j"$(nproc)" all && \
  make install

# Add to Python path
ENV PYTHONPATH=/root/caffe/python:$PYTHONPATH

# Set ~/caffe as working directory
WORKDIR /root/caffe
