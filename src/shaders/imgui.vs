#version 430 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec4 color;

layout (location = 0) out vec2 f_uv;
layout (location = 1) out vec4 f_color;

layout (binding = 0) uniform SceneData {
    mat4 clip_from_screen;
    mat4 mvp_matrices[];
};

void main() {
    f_uv = uv;
    f_color = color;
    gl_Position = clip_from_screen * vec4(position, 0.0, 1.0);
}