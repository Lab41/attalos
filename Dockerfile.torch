FROM  l41-nvidia-base

# Install git, apt-add-repository and dependencies for iTorch
RUN apt-get update && apt-get install -y \
    default-jre \
    git \
    hdf5-tools \
    ipython3 \
    libhdf5-serial-dev \
    libprotobuf-dev \
    libssl-dev \
    protobuf-compiler \
    python-pip \
    python-zmq \
    software-properties-common

# Install Jupyter Notebook for iTorch
RUN pip install notebook ipywidgets

# Run Torch7 installation scripts (dependencies only)
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && \
    cd /root/torch && \
    bash install-deps && \
    ./install.sh

# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Install cudnn manually as Torch opts for latest version
RUN cd /root && \
    git clone https://github.com/deepmind/torch-hdf5.git && \
    cd torch-hdf5 && \
    luarocks make hdf5-0-0.rockspec LIBHDF5_LIBDIR="/usr/lib/x86_64-linux-gnu/"

# add FindCUDA to remove annoying warnings
RUN wget https://github.com/hughperkins/FindCUDA/archive/v3.5-1.tar.gz -q -O FindCUDA-v3.5-1.tar.gz && \
    tar -zxf FindCUDA-v3.5-1.tar.gz && \
    cd FindCUDA-3.5-1 && \
    luarocks make rocks/findcuda-scm-1.rockspec && \
    rm /FindCUDA-v3.5-1.tar.gz

# Install additional DenseCap dependencies
RUN luarocks install torch
ENV CUDA_BIN_PATH=/usr/local/cuda-7.5
RUN luarocks install cutorch
RUN luarocks install nn
RUN luarocks install image
RUN luarocks install lua-cjson
RUN luarocks install https://raw.githubusercontent.com/soumith/cudnn.torch/R4/cudnn-scm-1.rockspec
RUN luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec
RUN luarocks install https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/torch-rnn-scm-1.rockspec
RUN luarocks install loadcaffe

# Install cuDNN
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
ENV CUDNN_VERSION 5
LABEL com.nvidia.cudnn.version="5"
RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
        libcudnn5 \
        libcudnn5-dev && \
    rm -rf /var/lib/apt/lists/*

# GPU acceleratation
RUN luarocks install cutorch
RUN luarocks install cunn
RUN luarocks install cudnn

# Set ~/torch as working directory
WORKDIR /root/torch
