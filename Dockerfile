FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

ENV TCNN_CUDA_ARCHITECTURES=75;80;86
ENV COLMAP_VERSION=3.7
ENV CMAKE_VERSION=3.21.0
ENV CERES_SOLVER_VERSION=2.0.0

RUN echo "Installing apt packages..." \
	&& export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	git \
	wget \
	ffmpeg \
	tk-dev \
	libxi-dev \
	libc6-dev \
	libbz2-dev \
	libffi-dev \
	libomp-dev \
	libssl-dev \
	zlib1g-dev \
	libcgal-dev \
	libgdbm-dev \
	libglew-dev \
	python3-dev \
	python3-pip \
	qtbase5-dev \
	checkinstall \
	libglfw3-dev \
	libeigen3-dev \
	libgflags-dev \
	libxrandr-dev \
	libopenexr-dev \
	libsqlite3-dev \
	libxcursor-dev \
	build-essential \
	libcgal-qt5-dev \
	libxinerama-dev \
	libboost-all-dev \
	libfreeimage-dev \
	libncursesw5-dev \
	libatlas-base-dev \
	libqt5opengl5-dev \
	libgoogle-glog-dev \
	libsuitesparse-dev \
	python3-setuptools \
	libreadline-gplv2-dev \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

RUN apt install python3

COPY ./requirements.txt ./

RUN echo "Installing pip packages..." \
	&& python3 -m pip install -U pip \
	&& pip3 --no-cache-dir install -r ./requirements.txt \
	&& pip3 --no-cache-dir install cmake==${CMAKE_VERSION} opencv-python \
	&& rm ./requirements.txt

RUN echo "Installing Ceres Solver ver. ${CERES_SOLVER_VERSION}..." \
	&& cd /opt \
	&& git clone https://github.com/ceres-solver/ceres-solver \
	&& cd ./ceres-solver \
	&& git checkout ${CERES_SOLVER_VERSION} \
	&& mkdir ./build \
	&& cd ./build \
	&& cmake ../ -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
	&& make -j \
	&& make install

RUN echo "Installing COLMAP ver. ${COLMAP_VERSION}..." \
	&& cd /opt \
	&& git clone https://github.com/colmap/colmap \
	&& cd ./colmap \
	&& git checkout ${COLMAP_VERSION} \
	&& mkdir ./build \
	&& cd ./build \
	&& cmake ../ \
	&& make -j \
	&& make install \
	&& colmap -h
	
RUN echo "Installing Instant-NGP..." \
	&& cd /opt \
	&& git clone --recursive https://github.com/NVlabs/instant-ngp \
	&& cd ./instant-ngp \
	&& cmake . -B build \
	&& cmake --build build --config RelWithDebInfo -j

COPY ./NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64.sh ./

RUN echo "Installing OptiX..." \
	&& mkdir /opt/optix \
	&& ./NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64.sh --skip-license --prefix=/opt/optix

COPY nerf_mesh_renderer /opt/nmr

RUN echo "Installing combined renderer..." \
	&& cd /opt/nmr \
	&& mkdir build \
	&& cd build \
	&& OptiX_INSTALL_DIR=/opt/optix cmake .. \
	&& make -j
