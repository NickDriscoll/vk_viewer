#version 430 core

layout (location = 0) in vec4 f_color;

layout (location = 0) out vec4 frag_color;

layout (binding = 0) uniform ColorBlock {
    vec4 tint;
};

void main() {
    frag_color = tint * f_color;
}