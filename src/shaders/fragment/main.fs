#version 430 core
#extension GL_EXT_nonuniform_qualifier: enable

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 frag_color;

#include "../frame_uniforms.sl"

layout(set = 0, binding = 1) uniform sampler2D global_textures[];

#include "../material_buffer.sl"

layout(push_constant) uniform Indices {
    uint material_idx;
};

void main() {
    vec3 normal = normalize(f_normal);
    float sun_contribution = max(0.1, dot(normal, sun_direction));

    Material my_mat = global_materials[material_idx];

    vec3 base_color = texture(global_textures[my_mat.color_map_index], f_uv).rgb;
    vec3 norm_color = normal * 0.5 + 0.5;

    vec3 final_color = sun_contribution * base_color;

    frag_color = vec4(final_color, 1.0);
}
