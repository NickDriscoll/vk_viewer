#version 430 core
#extension GL_EXT_nonuniform_qualifier: enable

layout (location = 0) in vec4 f_color;
layout (location = 1) in vec2 f_uv;

layout (location = 0) out vec4 frag_color;

layout (std140, set = 0, binding = 0) readonly uniform FrameData {
    mat4 clip_from_screen;
    mat4 clip_from_world;
    mat4 clip_from_view;
    mat4 view_from_world;
    float time;
};

layout(set = 0, binding = 1) uniform sampler2D global_textures[];

layout(push_constant) uniform Indices {
    uint tex_index;
};

void main() {
    frag_color = vec4(f_color.rgb * texture(global_textures[tex_index], f_uv).rgb, 1.0);
    //frag_color = vec4(f_color.rgb, 1.0);
    //frag_color = texture(global_textures[tex_index], f_uv);
}
