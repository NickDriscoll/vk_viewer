#version 430 core

layout (location = 0) in vec3 position;

layout (location = 0) out vec3 f_view_direction;

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
    f_view_direction = position;
    vec4 screen_space_pos = clip_from_skybox * vec4(position, 1.0);
    gl_Position = screen_space_pos.xyww;
}