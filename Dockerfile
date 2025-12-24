FROM ubuntu:24.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing
RUN apt-get upgrade -y

RUN apt-get install -y apt-transport-https 

RUN ln -fs /usr/share/zoneinfo/Australia/Brisbane /etc/localtime
RUN apt-get install -y tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install build-essential python3-venv python3-pip -y 
RUN apt-get install -y libtbbmalloc2 pocl-opencl-icd  
RUN python3 -m venv /venv
RUN chmod -R 777 /venv
ENV PATH="/venv/bin:$PATH"

RUN apt remove cmake -y
RUN pip install cmake --upgrade
RUN pip install conan && conan profile detect
RUN mkdir /app
RUN chmod 777 -R /root/.conan2/
RUN echo "tools.system.package_manager:mode = install" >> /root/.conan2/global.conf
RUN echo "core.cache:storage_path=/app/libs" >> /root/.conan2/global.conf

RUN apt-get install -y libopencv-dev 
RUN apt-get install -y libpng-dev 
RUN apt-get install -y optipng 
RUN apt-get install -y libva-dev

RUN apt-get install -y ocl-icd-opencl-dev opencl-headers 
#RUN apt-get install -y rocm-dev llvm clang
RUN apt install -y ntp ntpdate 

### BGS Library

WORKDIR /opt

# Changing INCUBATOR_VER will break the cache here
ARG INCUBATOR_VER=unknown

RUN apt install git -y 

RUN git clone https://github.com/andrewssobral/bgslibrary.git

ENV BGSLIBRARY_HOME=$PWD/bgslibrary

# Compile C++ library
WORKDIR /opt/bgslibrary/build
RUN cmake .. && make -j4 && make install

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

RUN pip install setuptools wheel

# Turn all shell scripts executable
WORKDIR /opt/bgslibrary
RUN chmod +x *.sh

#docker compose run --rm cpp-container bash

WORKDIR /app

