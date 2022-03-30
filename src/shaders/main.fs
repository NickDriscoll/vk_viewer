#version 430 core

layout (location = 0) in vec4 f_color;

layout (location = 0) out vec4 frag_color;

layout (set = 2, binding = 0) uniform sampler2D global_textures[];

void main() {
    //vec4 sampled_color = texture(global_textures[0], vec2(0.5));
    //frag_color = sampled_color;
    frag_color = f_color;
}
