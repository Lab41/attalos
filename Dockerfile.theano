FROM l41-nvidia-base
MAINTAINER Karl Ni

# Install Theano
RUN  pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
ENV THEANO_FLAGS='cuda.root=/path/to/cuda/root,device=gpu,floatX=float32'
ENV PATH=$PATH:/usr/local/cuda/bin
