FROM 903447430181.dkr.ecr.us-west-2.amazonaws.com/sagemaker-pytorch-base
LABEL maintainer="Dan R. Mbanga"
# vars
## Python version
ENV PYTHON_VERSION 3.5
ENV CUDNN_VERSION 7.0.3.11
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
## The version of this Docker container
ENV DANULAB_PYTORCH_IMAGE_VERSION 17.11
ENV DANULAB_PYTORCH_BUILD_VERSION 2.0

ENV PYTHONBUFFERED TRUE
# ENV CUDA_VISIBLE_DEVICES 0,1,2,3

ENV HOME_DIR /opt/ml/pytorch-neural-art

### Move files to the image 

WORKDIR ${HOME_DIR}

COPY . .

RUN chmod -R a+wx /opt && pip install flask_cors

ENTRYPOINT ["python3.5","algorithm/style_transfer.py" ]
