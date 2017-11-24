FROM nvidia/cuda:9.0-devel-ubuntu16.04
LABEL maintainer="Dan R. Mbanga"
# vars
## Python version
ENV PYTHON_VERSION 3.5
ENV CUDNN_VERSION 7.0.3.11
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
## The version of this Docker container
ENV DANULAB_PYTORCH_IMAGE_VERSION 17.11
ENV DANULAB_PYTORCH_BUILD_VERSION 2.0
## Dan'utils Anaconda channel for magma-cuda90. I had to build the package to support cuda9.0.
## Not yet available on the default Anaconda repository.
ENV DANULAB_ANACONDA_CHANNEL etcshadow

ENV PYTHONBUFFERED TRUE

ENV HOME_DIR /opt/ml/pytorch-neural-art
ENV ALGO_DIR ${HOME_DIR}/algorithm

RUN mkdir -p /opt/ml/input/config
RUN mkdir -p /opt/ml/input/data/train
RUN mkdir -p /opt/ml/input/data/eval
RUN mkdir -p /opt/ml/model
RUN mkdir -p /opt/ml/output 

### Move files to the image 

WORKDIR ${HOME_DIR}

COPY . .


###############################
# First, install apt-utils

RUN apt-get update

RUN apt-get install -y apt-utils

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN echo "deb-src http://us-east-1.ec2.archive.ubuntu.com/ubuntu/ xenial main restricted" >> /etc/apt/sources.list
RUN echo "deb-src http://us-east-1.ec2.archive.ubuntu.com/ubuntu/ xenial-updates main restricted" >> /etc/apt/sources.list

RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
    libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    zip unzip \
    htop \
    ca-certificates \
    libnccl2=2.0.5-3+cuda9.0 \
    libnccl-dev=2.0.5-3+cuda9.0 \
    libjpeg-dev \
    libpng-dev &&\
    rm -rf /var/lib/apt/lists/*

# Install Anaconda full. Not miniconda.
# https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh

RUN curl -o ~/miniconda-latest.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda-latest.sh && \
    ~/miniconda-latest.sh -b -p /opt/conda && \
    rm ~/miniconda-latest.sh && \
    /opt/conda/bin/conda install conda-build && \
    /opt/conda/bin/conda create -y --name pytorch-py35 python=${PYTHON_VERSION} \
    numpy pyyaml scipy ipython mkl jupyter && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/pytorch-py35/bin:$PATH

##### Installing magma-cuda90 from my Anaconda channel ##################################
# https://anaconda.org/etcshadow/magma-cuda90                                           #
#   To build from source (it takes a while!).                                           #
# WORKDIR /workspace                                                                    #
# RUN git clone https://github.com/pytorch/builder.git                                  #
# RUN cp -r builder/conda/magma-cuda90-2.2.0/ magma-cuda90                              #
# WORKDIR /workspace/magma-cuda90                                                       #
# RUN conda-build .                                                                     #
### Go to /opt/conda/conda-bld/magma-cuda90_1507361009645/  (your version might differ) #
# RUN conda install ./magma-cuda90-2.1.0-h865527f_5.tar.bz2                             #
#########################################################################################


RUN conda install --name pytorch-py35 -c ${DANULAB_ANACONDA_CHANNEL} magma-cuda90

WORKDIR /opt

RUN git clone --recursive https://github.com/pytorch/pytorch.git

WORKDIR /opt/pytorch

RUN git submodule update --init

#################################### BUILDING Pytorch for VOLTA on cuda9.0 ##################
# REF: https://github.com/torch/cutorch/blob/master/lib/THC/cmake/select_compute_arch.cmake #
# REF: https://en.wikipedia.org/wiki/CUDA                                                   #
# VOLTA has compute capabilities 7.0 and 7.1                                                #
# CUDA SDK 9.0 supports compute capability 3.0 through 7.x (Kepler, Maxwell, Pascal, Volta) #
#                                                                                           #
# Here we compile for:                                                                      #
#    - Kepler: 3.7 (AWS P2) +PTX, for the P2                                                #
#    - Maxwell: 5.0 5.2                                                                     #
#    - Jetson TX1: 5.3                                                                      #
#    - Pascal P100: 6.0                                                                     #
#    - Pascal GTX family: 6.1                                                               #
#    - Jetson TX2: 6.2                                                                      #
#    - Volta V100: 7.0+PTX (PTX = parallel Thread Execution): even faster!                  #
#############################################################################################

RUN TORCH_CUDA_ARCH_LIST="3.7+PTX 5.0 5.2 5.3 6.0 6.1+PTX 6.2 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .


# Working directory + examples

WORKDIR /opt

RUN git clone --recursive https://github.com/pytorch/vision.git && \
    cd vision && pip install -v .


RUN chmod -R a+wx /opt

ENTRYPOINT ["python3.5","/opt/ml/pytorch-neural-art/algorithm/style_transfer.py"]

