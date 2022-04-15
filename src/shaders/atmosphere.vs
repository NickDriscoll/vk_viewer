#version 430 core

layout (location = 0) in vec3 position;

layout (location = 0) out vec3 cube_sampling_vector;

layout (std140, set = 0, binding = 0) readonly uniform FrameData {
    mat4 clip_from_screen;
    mat4 clip_from_world;
    mat4 clip_from_view;
    mat4 view_from_world;
    mat4 clip_from_skybox;
    vec3 sun_direction;
    float time;
};

void main() {
    
}