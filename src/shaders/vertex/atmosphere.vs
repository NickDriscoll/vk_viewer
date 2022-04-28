#version 430 core

layout (location = 0) in vec3 position;

layout (location = 0) out vec3 f_view_direction;

#include "../frame_uniforms.sl"

void main() {
    f_view_direction = position;
    vec4 screen_space_pos = clip_from_skybox * vec4(position, 1.0);
    gl_Position = screen_space_pos.xyww;
}