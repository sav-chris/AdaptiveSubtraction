FROM ubuntu:22.04 AS base

RUN apt-get update
RUN apt-get upgrade -y
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
RUN rm -f /root/.conan2/global.conf

