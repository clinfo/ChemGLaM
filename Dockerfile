# Use the official CUDA 12.1.1 image as a parent image
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    gnupg2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV MINICONDA_VERSION=latest
ENV MINICONDA_PREFIX=/opt/miniconda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p ${MINICONDA_PREFIX} \
    && rm /tmp/miniconda.sh

# Set the environment variable
ENV PATH="${MINICONDA_PREFIX}/bin:${PATH}"

# Create a new conda environment
RUN conda create -n chemglam python=3.11

# Install the required packages
RUN /bin/bash -c "source activate chemglam && \
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install transformers==4.46.3 && \
    pip install lightning==2.4.0 && \
    pip install peft==0.13.2 && \
    pip install pandas==2.2.3 && \
    pip install scikit-learn==1.5.2 && \
    pip install -U wandb>=0.12.0 && \
    conda install -c conda-forge rdkit=2024.9.2"

# Set the working directory
WORKDIR /workspace

# Set the default entrypoint
ENTRYPOINT ["bash", "-c", "source activate chemglam && exec \"$@\"", "--"]

ENV HF_HOME=/workspace/.cache
ENV MPLCONFIGDIR=/workspace/.cache/matplotlib
RUN mkdir -p $MPLCONFIGDIR && chmod -R 777 $MPLCONFIGDIR
ENV WANDB_MODE=disabled

# Set the default command to bash
CMD ["python", "train.py", "-c", "config/benchmark/bindingdb_cv0.json"]

