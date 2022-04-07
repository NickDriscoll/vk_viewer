#version 430 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 tangent;
layout (location = 2) in vec3 bitangent;
layout (location = 3) in vec3 normal;
layout (location = 4) in vec2 uv;

layout (location = 0) out vec4 f_color;
layout (location = 1) out vec2 f_uv;
layout (location = 2) out int instance_id;

layout (std140, set = 0, binding = 0) readonly buffer FrameData {
    mat4 clip_from_screen;
    mat4 mvp_matrices[];
};

layout (std140, set = 2, binding = 0) readonly uniform InstanceData {
    ivec4 material_indices[];
};

void main() {
    f_color = vec4(normal * 0.5 + 0.5, 1.0);
    f_uv = uv;
    instance_id = gl_InstanceIndex;
    gl_Position = mvp_matrices[gl_InstanceIndex] * vec4(position, 1.0);
}
