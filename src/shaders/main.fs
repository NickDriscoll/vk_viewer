#version 430 core
#extension GL_EXT_nonuniform_qualifier: enable

layout (location = 0) in vec4 f_color;

layout (location = 0) out vec4 frag_color;

layout(set = 0, binding = 1) uniform sampler2D global_textures[];

void main() {
    //vec4 sampled_color = texture(global_textures[0], vec2(0.5));
    //frag_color = sampled_color;
    //vec4 c = texture(global_textures[0], vec2(0.0));
    frag_color = f_color;
}
