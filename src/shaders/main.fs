#version 430 core
#extension GL_EXT_nonuniform_qualifier: enable

layout (location = 0) in vec4 f_color;
layout (location = 1) in vec2 f_uv;

layout (location = 0) out vec4 frag_color;

layout(set = 0, binding = 1) uniform sampler2D global_textures[];

layout(push_constant) uniform Indices {
    uint tex_index;
};

void main() {
    frag_color = f_color * texture(global_textures[tex_index], f_uv);
}
