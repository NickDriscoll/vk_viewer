#version 430 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec4 color;

layout (location = 0) out vec2 f_uv;
layout (location = 1) out vec4 f_color;

layout (std140, set = 0, binding = 0) readonly uniform FrameData {
    mat4 clip_from_screen;
    mat4 clip_from_world;
    mat4 clip_from_view;
    mat4 view_from_world;
};

void main() {
    f_uv = uv;
    f_color = color;
    gl_Position = clip_from_screen * vec4(position, 0.0, 1.0);
}