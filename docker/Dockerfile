FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
LABEL maintainer="Yaojie Lu"
LABEL repository="uie"

ARG PYTHON_VERSION=3.8
ARG CONDA_ENV=env

# https://developer.nvidia.com/zh-cn/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt update && \
        apt-get install -y --no-install-recommends \
        build-essential ca-certificates git curl wget && \
        apt autoremove -y && \
        apt-get clean && \
        rm -rf /root/.cache && \
        rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/envs/env/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sb
in:/bin

RUN curl -o miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
        /bin/bash miniconda3.sh -b -p /opt/conda && rm miniconda3.sh && \
        conda update -n base -c defaults conda && \
        conda create -n ${CONDA_ENV} python=${PYTHON_VERSION} && \
        conda clean -ay && \
        echo "source activate ${CONDA_ENV}" >> ~/.bashrc

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir mkl && \
    python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && \
    python3 setup.py install && \
    python3 -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

COPY ./requirements.txt .
RUN python3 -m pip install -r ./requirements.txt

CMD ["/bin/bash"]
