#version 430 core
#extension GL_EXT_nonuniform_qualifier: enable

layout (location = 0) in vec2 f_uv;
layout (location = 1) in vec4 f_color;

layout (location = 0) out vec4 frag_color;

layout(set = 1, binding = 0) uniform sampler2D global_textures[];

layout(push_constant) uniform Constants {
    uint font_atlas_index;
};

void main() {
    frag_color = f_color * texture(global_textures[font_atlas_index], f_uv).r;
}