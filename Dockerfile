FROM gcc:latest

# docker build -t connnortbot/caitlyn-mcrt:TAG_NAME .
# docker push connortbot/caitlyn-mcrt:TAG_NAME

# This is a test to push to docker and github

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

# Install Embree
ARG EMBREE_VERSION=4.3.0
ADD https://github.com/embree/embree/releases/download/v${EMBREE_VERSION}/embree-${EMBREE_VERSION}.x86_64.linux.tar.gz /opt
RUN cd /opt && \
    tar xzf embree-${EMBREE_VERSION}.x86_64.linux.tar.gz
RUN echo "source embree-${EMBREE_VERSION}.x86_64.linux/embree-vars.sh" >> /root/.profile

WORKDIR /app
