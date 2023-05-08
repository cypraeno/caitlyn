# The Caitlyn Renderer
Caitlyn is a ray-tracing render engine built in C++ and aimed to provide higher-level graphics for 3D film animation done in Godot. It is meant for the visual style originally developed and inspired by the Odd Tales’ “The Last Night”. It is also the main visual style used in ‘Caitlyn’, a film I (Connor) created in 2021.

## Contributing and Current Status
If you wish to contribute to this project, please email me at ctloi@uwaterloo.ca and I'll add you to RenderTeam.
We are currently following the fantastic work of [Peter Shirley and his team](https://raytracing.github.io/).

**CURRENT GOALS:**
- Parallelism and GPU Acceleration via Nvidia CUDA [ACTIVE]
- Dependency and compilation automation through CMake and vcpkg [ACTIVE]
- Peter Shirley Book 2
- Denoising
- Plane textures and vec4 transparency

Note: ACTIVE indicates a branch is open for this right now.


## Future README
THE BELOW TEXT IS DESCRIBING THE GOAL FINAL VERSION, NOT THE CURRENT VERSION

The goal of Caitlyn was to leverage my long-time usage and knowledge of Godot and being able to convert it into high-quality rendering animation. No more OpenGL and GLES3!
It does this by converting .tscn files into .cait files, the standard scene formatter used in Caitlyn. Of course, it is very daunting to write a full converter of everything (as many things in .tscn files are NOT convertible to an animation setting) so Caitlyn’s main goal is only the necessary parts of the visual style which I’ve dubbed ‘pixellax’. 
Planes w/ textures
Cubes w/ textures
Animations
The above parts are essentially every varying type involved in pixellax animation.


A lot of the work done on Caitlyn is inspired or straight up following the fantastic work of Peter Shirley and his team. You can find their detailed guide below:
https://raytracing.github.io/
