FROM gcc:latest

# docker build -t caitlyn-mcrt .
# docker tag caitlyn-mcrt connortbot/caitlyn-mcrt
# docker push connortbot/caitlyn-mcrt


# Install required packages
RUN apt-get update && \
    apt-get install -y libsdl2-dev curl zip unzip tar git wget

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

# Install vcpkg
RUN git clone https://github.com/microsoft/vcpkg && \
    ./vcpkg/bootstrap-vcpkg.sh && \
    ./vcpkg/vcpkg integrate install

COPY . /app
WORKDIR /app
