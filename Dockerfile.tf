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

# Install Tensorflow
# --ignore-installed is required when using Anaconda per: https://github.com/tensorflow/tensorflow/issues/622#issuecomment-170309570
RUN pip install --upgrade --ignore-installed https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl

EXPOSE 8888
