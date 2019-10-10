ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG BASE_CONTAINER=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu16.04
FROM $BASE_CONTAINER
ENV LOG_PATH=/opt/log/my_logs
ENV ANACONDA_DIR=/opt/anaconda
RUN mkdir -p $LOG_PATH
WORKDIR /app
RUN mkdir /usr/share/bind_mount \
 && mkdir /usr/share/bind_mount/data \
 && mkdir /usr/share/bind_mount/scripts 

# Update and Install basic commands + clear cache
RUN apt-get update \
 && apt-get install -y \
	sudo \
	wget \
	git \
	curl \
	unzip \
	bzip2 \
	screen \
 &&  rm -rf /var/lib/apt/lists/*

# Install anaconda
RUN curl -o ~/anaconda.sh -0 https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh \
 && chmod +x ~/anaconda.sh \ 
 && ~/anaconda.sh -b -p $ANACONDA_DIR \
 && rm ~/anaconda.sh

# Add anaconda to the path
ENV PATH $ANACONDA_DIR/bin:${PATH}
 
# Install pytorch
RUN conda install -y pandas \
	opencv \
	termcolor \	
	pytorch \
	torchvision \
	cudatoolkit=${CUDA_VERSION} -c pytorch \
 && conda clean -ya

# Install tensorboard
RUN pip install --upgrade torch \
 && pip install tb-nightly \
 && pip install future \
 && pip install easydict \
 && pip install tensorly \
 && pip install torchnet \
 && pip install tensorflow
# Launch Tensorboard
#ENTRYPOINT ["tensorboard", "--logdir", "$LOG_PATH"]
