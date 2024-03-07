FROM mcr.microsoft.com/devcontainers/base:jammy

# install ubuntu packages
RUN apt-get update && apt-get install -y \
    git \
    clang \
    cmake

# install c++ libraries
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libeigen3-dev