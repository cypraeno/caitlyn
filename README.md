<h1 align="center">The Caitlyn Renderer :camera:</h1>
<p align="center"><img width="600" alt="Render1" src="https://github.com/cypraeno/caitlyn/assets/25397938/9f93e7a7-37d0-43e4-bea1-e81859f75f00"></p>


Caitlyn is an in-development Monte Carlo ray tracer built in C++ by a team of students from the University of Waterloo and Wilfrid Laurier University. Caitlyn is actively working towards being a beautiful tool for bridging the accuracy and lighting of raytracing and the breathtaking visuals of pixel art (see our [portfolio](#our-portfolio) and Odd Tales' "The Last Night"!)

_Interested in getting involved? Contact [Connor Loi](ctloi@uwaterloo.ca) or [Samuel Bai](sbai@uwaterloo.ca)._

## Table of Contents
- [Quick Start Guide](#quick-start-guide)
- [Our Portfolio](#our-portfolio)
- [In-depth Docs](#docs)
    - [Writing Scenes](#writing-scenes)
    - [Rendering](#rendering)
- [Contribute](#contribute)

## Quick Start Guide
Caitlyn MCRT is built on Debian 12. It may work on other distros, but we recommend simply pulling our Docker container with the `connortbot/caitlyn-mcrt` repository. We recommend cloning the repository, mounting a volume, initializing our validation submodule, and compiling.

### Setup
Before continuing:
- Install `Docker Desktop`.
- Pull the latest `docker pull connortbot/caitlyn-mcrt:base-vX.X.X`

### Build
You may pull the repository from within the container or mount a volume. Either works!
Run `cmake -B build/ -S .` to create files in the `build` folder. `cd build`, and `make`. Don't forget to initialize the submodules.

### Basic Rendering
Caitlyn renders scenes from our custom filetype `.csr`. By default, the `caitlyn` executable will read the scene from a `scene.csr` file, so you need to have one before running. In this guide, we'll just run the `example.csr`, which you can copy from [here](https://github.com/cypraeno/csr-schema/blob/main/examples/example.csr).

To learn how to write CSR files, check out the [Basic Guide](https://github.com/cypraeno/csr-schema/blob/main/docs/basic-guide.md).

Caitlyn has a user-friendly command line interface, allowing you to customize samples, depth, type of multithreading, and more. Once you have the executable, you can run `./caitlyn --help` to see all the options at your disposal.

Let's render a PNG file of the example scene! Ensure that you have your CSR file in the same directory.
```
./caitlyn -i example.csr -t png -r 600 600
```
This will read the scene from `example.csr` and output as a `png`.
And now you have your first caitlyn-rendered scene!

## Our Portfolio
![image](https://github.com/cypraeno/caitlyn/assets/25397938/38ad0953-1c29-4ae6-aead-fa2523706b3b)

## Docs

### Writing Scenes
As mentioned in the `Quick Start`, `caitlyn` will read and build scenes via CSR files. The CSR [Basic Guide](https://github.com/cypraeno/csr-schema/blob/main/docs/basic-guide.md) covers everything from creating objects to custom materials.

### Rendering
To see all the options available to `caitlyn`, run:
```
./caitlyn --help
```
Flags like `--samples` and `--depth` control the amount of time spent on the render. If you are unfamiliar with rendering concepts, `samples` refer to the amount of rays traced per pixel, which decreases the noise of a render as it increases. `depth` refers to the amount of times a ray is simulated to "bounce" around the scene, allowing for realism in reflections, emissives, etc.

Sometimes, CSR files will have features not supported in your version of `caitlyn`. You can check this with the version indicator at the top of the CSR file and with `./caitlyn --version`.

For users who have a better understanding of their computer's resources, the `--threads` and `--vectorization` flags control the use of more efficient architecture. While `threads` dictate the amount of CPU threads to split the workloads on, the `vectorization` flag will dictate the type of SIMD batching. `[NONE|SSE|AVX|AVX512]`.


## Contribute
For contribution or general inquiries, please email one of us at [Connor Loi](ctloi@uwaterloo.ca) or [Samuel Bai](sbai@uwaterloo.ca).

The people the made it happen:

<div align="center">
<a href="https://github.com/connortbot">Connor Loi</a>,
<a href="https://github.com/haenlonns">Samuel Bai</a>,
<a href="https://github.com/Saai151">Saai Arora</a>,
<a href="https://github.com/dan-the-man639">Danny Yang</a>,
<a href="https://github.com/ASharpMarble">Jonathan Wang</a>,
<a href="https://github.com/18gen">Gen Ichihashi</a>,
<a href="https://github.com/maxtan84">Max Tan</a>,
<a href="https://github.com/rickyhuangjh">Ricky Huang</a>,
</div>
