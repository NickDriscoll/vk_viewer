#version 430 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 tangent;
layout (location = 2) in vec3 bitangent;
layout (location = 3) in vec3 normal;
layout (location = 4) in vec2 uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

layout (std140, set = 0, binding = 0) readonly uniform FrameData {
    mat4 clip_from_screen;
    mat4 clip_from_world;
    mat4 clip_from_view;
    mat4 view_from_world;
    mat4 clip_from_skybox;
    vec3 sun_direction;
    float time;
};

layout (std140, set = 0, binding = 2) readonly buffer InstanceData {
    mat4 model_matrices[];
};

void main() {
    mat4 model_matrix = model_matrices[gl_InstanceIndex];
    mat4 normal_matrix = transpose(mat4(inverse(mat3(model_matrix))));
    vec3 world_normal = (normal_matrix * vec4(normal, 0.0)).xyz;
    vec3 world_position = (model_matrix * vec4(position, 1.0)).xyz;
    f_position = world_position;
    f_normal = world_normal;
    f_uv = uv;
    gl_Position = clip_from_world * vec4(world_position, 1.0);
}
