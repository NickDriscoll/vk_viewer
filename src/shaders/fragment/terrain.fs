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
    uint first_material_idx;
};

void main() {
    vec3 tangent = normalize(f_tangent);
    vec3 bitangent = normalize(f_bitangent);
    vec3 normal = normalize(f_normal);
    mat3 TBN = mat3(tangent, bitangent, normal);

    Material material1 = global_materials[first_material_idx];
    Material material2 = global_materials[first_material_idx + 1];
    vec3 base_color1 = texture(global_textures[material1.color_map_index], f_uv).rgb;
    vec3 base_color2 = texture(global_textures[material2.color_map_index], f_uv).rgb;
    vec3 sampled_normal1 = normalize(2.0 * texture(global_textures[material1.normal_map_index], f_uv).xyz - 1.0);
    vec3 sampled_normal2 = normalize(2.0 * texture(global_textures[material2.normal_map_index], f_uv).xyz - 1.0);

    float mix_factor = max(0.0, dot(normal, vec3(0.0, 0.0, 1.0)));
    mix_factor = smoothstep(0.60, 0.70, mix_factor);

    vec3 world_normal = TBN * mix(sampled_normal2, sampled_normal1, mix_factor);
    //vec3 world_normal = TBN * sampled_normal1;
    float sun_contribution = max(0.1, dot(world_normal, sun_direction) + 0.1);

    vec3 final_color = mix(base_color2, base_color1, mix_factor);    
    final_color *= sun_contribution;

    frag_color = vec4(final_color, 1.0);
}
