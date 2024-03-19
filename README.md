<h1 align="center">The Caitlyn Renderer :camera:</h1>
<p align="center"><img width="600" alt="Render1" src="https://github.com/Astro-Monkeys/caitlyn/assets/25397938/8f088c62-47e1-432d-9c12-9a198214d6b0"></p>

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
Caitlyn MCRT is built on Debian 12. It may work on other distros, but we recommend simply pulling our Docker container.

### Setup
Before continuing:
- Install `Docker Desktop`.
- Pull the latest `docker pull connortbot/caitlyn-mcrt:base-vX.X.X`

### Build
You may pull the `stable` repository from within the container or mount a volume. Either works!
Run `cmake -B build/ -S .` to create files in the `build` folder. `cd build`, and `make`.

### Basic Rendering
Caitlyn renders scenes from our custom filetype `.csr`. By default, the `caitlyn` executable will read the scene from a `example.csr` file, so you need to have one before running. In this guide, we'll just run the `example.csr` 

To learn how to write CSR files, check out the [Basic Guide](https://github.com/cypraeno/csr-schema/blob/main/basic-guide.md).

Caitlyn has a user-friendly command line interface, allowing you to customize samples, depth, type of multithreading, and more. 

## Our Portfolio

## Docs

### Writing Scenes

### Rendering

## Contribute
For contribution or general inquiries, please email one of us at [Connor Loi](ctloi@uwaterloo.ca) or [Samuel Bai](sbai@uwaterloo.ca).

Behind every pixel rendered is our incredible developers who have contributed their dedication and creativity to Caitlyn.
<div align="center">
<a href="https://github.com/connortbot">Connor Loi</a>,
<a href="https://github.com/haenlonns">Samuel Bai</a>,
<a href="https://github.com/Saai151">Saai Arora</a>,
<a href="https://github.com/dan-the-man639">Danny Yang</a>,
<a href="https://github.com/ASharpMarble">Jonathan Wang</a>,
<a href="https://github.com/18gen">Gen Ichihashi</a>,
<a href="https://github.com/18gen">Max Tan</a>,
<a href="https://github.com/rickyhuangjh">Ricky Huang</a>,
<a href="https://github.com/daniel-su1">Daniel Su</a>
</div>
