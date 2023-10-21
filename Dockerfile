FROM gcc:latest
<<<<<<< HEAD
=======

>>>>>>> ca4dd0345c870a86cbe8801b0a3ebae4014b7ed1
# approved
# Install required packages
RUN apt-get update && \
    apt-get install -y curl zip unzip tar git wget

# Install CMake
ARG CMAKE_VERSION=3.26.0
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar -xzvf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap && \
    make && \
    make install && \
    cd .. && \
    rm -rf cmake-${CMAKE_VERSION}*

# -rf : r recursive f force
# -xzvf : x extract -z gunzip (compress as extracting) -v verbose (list out extracts) -f file (follow cmake instructions)
# bootstrap : make ready for compilation

COPY . /app
WORKDIR /app
