#version 430 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

layout (location = 0) out vec4 f_color;

layout (set = 0, binding = 0) uniform TransformBlock {
    mat4 mvp_matrices[];
};

layout(push_constant) uniform Constants {
    uint model_index;

};

void main() {
    f_color = vec4(color, 1.0);
    gl_Position = mvp_matrices[1] * vec4(position, 1.0);
}
