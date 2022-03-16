#version 430 core

layout (location = 0) in vec4 f_color;

layout (location = 0) out vec4 frag_color;

void main() {
    frag_color = f_color;
}