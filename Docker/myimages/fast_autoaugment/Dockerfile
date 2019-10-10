ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG BASE_CONTAINER=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu16.04
FROM $BASE_CONTAINER
ENV LOG_PATH=/opt/log/my_logs
ENV ANACONDA_DIR=/opt/anaconda
ENV CUDNN_VERSION=7.6.0.64-1+cuda10.0
ENV NCCL_VERSION=2.4.7-1+cuda10.0
RUN mkdir -p $LOG_PATH
WORKDIR /app
RUN mkdir /usr/share/bind_mount \
 && mkdir /usr/share/bind_mount/data \
 && mkdir /usr/share/bind_mount/scripts 

# Update and Install basic commands + clear cache
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
	librdmacm1 \
        libibverbs1 \
	sudo \
	unzip \
	bzip2 \
	screen \
 &&  rm -rf /var/lib/apt/lists/*
        
# Install ssh
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

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


# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_PYTORCH=1 \
         pip install --no-cache-dir horovod && \
    ldconfig

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config


# Install tensorboard and Flops library
RUN pip install --upgrade torch \
 && pip install tb-nightly \
 && pip install future \
 && pip install easydict \
 && pip install tensorly
# Launch Tensorboard
#ENTRYPOINT ["tensorboard", "--logdir", "$LOG_PATH"]
