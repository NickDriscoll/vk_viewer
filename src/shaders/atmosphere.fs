#version 430 core

layout (location = 0) in vec3 cube_sampling_vector;

layout (location = 0) out vec4 frag_color;

void main() {
    frag_color = vec4(0.0, 1.0, 0.0, 1.0);
}