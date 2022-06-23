# Download base image intel (oneapi with hpcket on ubuntu 18.04)
FROM intel/oneapi-hpckit:latest
LABEL maintainer="joseph.parker@ukaea.uk"
LABEL version="0.0.1"
LABEL description="Image for running NESO, based on Intel's oneapi running on Ubuntu 18.04 LTS"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# NB: Do NOT install cmake with apt. The version provided by intel is newer
# than the version in Ubuntu 18.04.

RUN apt-get update && apt install -y \
	software-properties-common\
	git \
	vim \
	rsync \
	libnetcdf-dev \
	libnetcdff-dev \
	python3 \
	libblas-dev \
	libhwloc-dev \
	liblapack-dev \
	netcdf-bin \
	bison

RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get install -y gcc-11 g++-11

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 20
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 20
