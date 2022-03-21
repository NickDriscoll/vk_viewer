#version 430 core

layout (location = 0) in vec2 f_uv;
layout (location = 1) in vec4 f_color;

layout (location = 0) out vec4 frag_color;

layout (binding = 0) uniform sampler2D font_atlas;

void main() {
    frag_color = f_color * texture(font_atlas, f_uv).r;
}