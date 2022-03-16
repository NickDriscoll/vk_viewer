#version 430 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec3 color;

layout (location = 0) out vec4 f_color;

layout (binding = 0) uniform TransformBlock {
    mat4 mvp;
};

void main() {
    f_color = vec4(color, 1.0);
    gl_Position = vec4(position, 0.0, 1.0);
}