#version 430 core
#extension GL_EXT_nonuniform_qualifier: enable

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec3 f_tangent;
layout (location = 4) in vec3 f_bitangent;

layout (location = 0) out vec4 frag_color;

#include "../frame_uniforms.sl"

layout(set = 0, binding = 1) uniform sampler2D global_textures[];

#include "../material_buffer.sl"

layout(push_constant) uniform Indices {
    uint material_idx;
};

void main() {
    vec3 tangent = normalize(f_tangent);
    vec3 bitangent = normalize(f_bitangent);
    vec3 normal = normalize(f_normal);
    mat3 TBN = mat3(tangent, bitangent, normal);

    Material my_mat = global_materials[material_idx];

    vec3 base_color = texture(global_textures[my_mat.color_map_index], f_uv).rgb;
    vec3 sampled_normal = 2.0 * texture(global_textures[my_mat.normal_map_index], f_uv).xyz - 1.0;
    vec3 world_normal = TBN * sampled_normal;
    float sun_contribution = max(LIGHTING_MIN, dot(world_normal, sun_direction));

    vec3 final_color = sun_contribution * base_color;

    frag_color = vec4(final_color, 1.0);
}
