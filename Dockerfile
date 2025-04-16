# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61"
ARG NERFSTUDIO_VERSION="main"

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder
ARG CUDA_ARCHITECTURES
ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# ------------------------------------------------------------------------
# 1. Basic system setup and dependencies
# ------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
    git wget ninja-build build-essential \
    libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev \
    python3.10-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/nerfstudio-project/nerfstudio.git /tmp/nerfstudio
RUN git clone --branch CLI --recursive https://github.com/AndreaAmorosini/full_pipe_mast3r_nerfstudio.git /opt/full_pipe_mast3r_nerfstudio

# ------------------------------------------------------------------------
# 2. Install a newer CMake
# ------------------------------------------------------------------------
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh && \
    chmod u+x /tmp/cmake-install.sh && \
    mkdir /opt/cmake-3.28.3 && \
    /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.28.3 && \
    rm /tmp/cmake-install.sh && \
    ln -s /opt/cmake-3.28.3/bin/* /usr/local/bin

# ------------------------------------------------------------------------
# 3. Build and install GLOMAP
# ------------------------------------------------------------------------
RUN git clone https://github.com/colmap/glomap.git && \
    cd glomap && \
    git checkout "1.0.0" && \
    mkdir build && cd build && \
    mkdir -p /build && \
    cmake .. -GNinja "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    -DCMAKE_INSTALL_PREFIX=/build/glomap && \
    ninja install -j1 && \
    cd ~ && rm -rf glomap

# ------------------------------------------------------------------------
# 4. Build and install COLMAP
# ------------------------------------------------------------------------
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout "3.9.1" && \
    mkdir build && cd build && \
    mkdir -p /build && \
    cmake .. -GNinja "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    -DCMAKE_INSTALL_PREFIX=/build/colmap && \
    ninja install -j1 && \
    cd ~ && rm -rf colmap

# ------------------------------------------------------------------------
# 5. Python: upgrade pip and install PyTorch
# ------------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip 'setuptools<70.0.0'
RUN pip install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 'numpy<2.0.0' \
    --extra-index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------------------
# 6. (Optional) hloc and tiny-cuda-nn for Nerfstudio, if needed
# ------------------------------------------------------------------------
RUN export GIT_TERMINAL_PROMPT=0 && \
    git clone --branch v1.4 --recursive https://github.com/cvg/Hierarchical-Localization.git /opt/hloc && \
    cd /opt/hloc && python3.10 -m pip install --no-cache-dir . && \
    cd ~ && rm -rf /opt/hloc

RUN export GIT_TERMINAL_PROMPT=0 && \
    export TCNN_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" && \
    pip install --no-cache-dir "git+https://github.com/NVlabs/tiny-cuda-nn.git@b3473c81396fe927293bdfd5a6be32df8769927c#subdirectory=bindings/torch"

RUN pip install --no-cache-dir pycolmap==0.6.1 pyceres==2.1 omegaconf==2.3.0

# ------------------------------------------------------------------------
# 6.5. Install Miniconda and create a conda environment for mast3r packages
# ------------------------------------------------------------------------
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh
# Make conda available
ENV PATH=/opt/miniconda/bin:$PATH
RUN conda update -n base -c defaults conda -y && \
    conda create -y -n mast3r_env python=3.10

# ------------------------------------------------------------------------
# 7. MAST3R installation using the conda environment
# ------------------------------------------------------------------------
# a) Clone MAST3R (recursive to grab submodules)
RUN conda run -n mast3r_env git clone --branch mast3r_sfm --recursive https://github.com/naver/mast3r.git /opt/mast3r

# b) Install main Python dependencies from the repo inside the conda environment
WORKDIR /opt/mast3r
RUN conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda run -n mast3r_env pip install --no-cache-dir -r requirements.txt
# If you need optional packages, uncomment or adapt the following line:
RUN conda run -n mast3r_env pip install --no-cache-dir -r dust3r/requirements.txt && \
    conda run -n mast3r_env pip install --no-cache-dir -r dust3r/requirements_optional.txt

# c) Additional dependencies and C extensions
RUN conda run -n mast3r_env pip install --no-cache-dir pyaml faiss-gpu
RUN conda run -n mast3r_env pip install --no-cache-dir cython

RUN conda run -n mast3r_env git clone https://github.com/jenicek/asmk /opt/mast3r/asmk

# d) Compile and install ASMK
WORKDIR /opt/mast3r/asmk/cython
RUN conda run -n mast3r_env cythonize *.pyx
WORKDIR /opt/mast3r/asmk
RUN conda run -n mast3r_env pip install --no-cache-dir "numpy<2.0.0"
RUN conda run -n mast3r_env pip install --no-cache-dir .

# e) (Optional) Compile the CUDA kernels for RoPE (CroCo v2).
#    Uncomment the lines below if needed:
# WORKDIR /opt/mast3r/dust3r/croco/models/curope
# RUN conda run -n mast3r_env python3 setup.py build_ext --inplace

# ------------------------------------------------------------------------
# 8. Install gsplat + Nerfstudio
# ------------------------------------------------------------------------
WORKDIR /
RUN pip install --no-cache-dir git+https://github.com/cansik/sharp-frame-extractor.git@1.6.6
# The following steps remain as before:
RUN export GIT_TERMINAL_PROMPT=0 && \
    export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export MAX_JOBS=4 && \
    GSPLAT_VERSION="$(sed -n 's/.*gsplat==\s*\([^," '"'"']*\).*/\1/p' /tmp/nerfstudio/pyproject.toml)" && \
    pip install --no-cache-dir git+https://github.com/nerfstudio-project/gsplat.git@v${GSPLAT_VERSION} && \
    pip install --no-cache-dir /tmp/nerfstudio 'numpy<2.0.0' && \
    pip install --no-cache-dir git+https://github.com/KevinXu02/splatfacto-w && \
    rm -rf /tmp/nerfstudio

# ------------------------------------------------------------------------
# 9. Permissions cleanup
# ------------------------------------------------------------------------
RUN chmod -R go=u /usr/local/lib/python3.10 && chmod -R go=u /build
RUN chmod -R go=u /usr/local/lib/python3.10 && chmod -R go=u /build

# ------------------------------------------------------------------------
# 10. Runtime stage
# ------------------------------------------------------------------------
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime
ARG CUDA_ARCHITECTURES
ARG NVIDIA_CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.source="https://github.com/nerfstudio-project/nerfstudio"
LABEL org.opencontainers.image.licenses="Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}"
LABEL org.opencontainers.image.documentation="https://docs.nerf.studio/"

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
    git libboost-filesystem1.74.0 libboost-program-options1.74.0 libc6 libceres2 libfreeimage3 \
    libgcc-s1 libgl1 libglew2.2 libgoogle-glog0v5 libqt5core5a libqt5gui5 libqt5widgets5 \
    python3.10 python3.10-dev build-essential python-is-python3 ffmpeg \
    libopencv-core4.5d libtbb2 libjpeg-dev libpng-dev libtiff-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages and binaries from builder stage.
COPY --from=builder /build/colmap/ /usr/local/
COPY --from=builder /build/glomap/ /usr/local/
COPY --from=builder /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /opt/full_pipe_mast3r_nerfstudio/ /full_pipe_mast3r_nerfstudio

# Copy the Miniconda installation (and thus the mast3r_env) into the runtime stage.
COPY --from=builder /opt/miniconda /opt/miniconda
ENV PATH=/opt/miniconda/bin:$PATH

RUN /bin/bash -c 'ns-install-cli --mode install'
RUN /bin/bash -c "conda init"
RUN /bin/bash -c "pip install 'fastapi[standard]' minio boto3"

WORKDIR /full_pipe_mast3r_nerfstudio
CMD ["/bin/bash", "-l"]
# EXPOSE 8000

# CMD ["fastapi", "run", "main.py", "--port", "8090", "--host", "0.0.0.0"]
