#version 430 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 tangent;
layout (location = 2) in vec3 bitangent;
layout (location = 3) in vec3 normal;
layout (location = 4) in vec2 uv;

layout (location = 0) out vec4 f_color;
layout (location = 1) out vec2 f_uv;

layout (std140, set = 0, binding = 0) readonly uniform FrameData {
    mat4 clip_from_screen;
    mat4 clip_from_world;
    mat4 clip_from_view;
    mat4 view_from_world;
};

layout (std140, set = 0, binding = 2) readonly buffer InstanceData {
    mat4 model_matrices[];
};

void main() {
    f_color = vec4(normal * 0.5 + 0.5, 1.0);
    f_uv = uv;
    gl_Position = clip_from_world * model_matrices[gl_InstanceIndex] * vec4(position, 1.0);
}
