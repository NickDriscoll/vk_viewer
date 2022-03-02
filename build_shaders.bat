@echo off

glslangValidator.exe -V -S vert -o ./shaders/main_vert.spv ./src/shaders/main.vs
glslangValidator.exe -V -S frag -o ./shaders/main_frag.spv ./src/shaders/main.fs