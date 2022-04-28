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
    uint first_material_idx;
};

void main() {
    vec3 normal = normalize(f_normal);
    float sun_contribution = max(0.1, dot(normal, sun_direction));

    Material material1 = global_materials[first_material_idx];
    Material material2 = global_materials[first_material_idx + 1];
    vec3 base_color1 = texture(global_textures[material1.color_map_index], f_uv).rgb;
    vec3 base_color2 = texture(global_textures[material2.color_map_index], f_uv).rgb;
    vec3 norm_color = normal * 0.5 + 0.5;

    float mix_factor = max(0.0, dot(normal, vec3(0.0, 0.0, 1.0)));
    mix_factor = smoothstep(0.60, 0.70, mix_factor);
    vec3 final_color = mix(base_color2, base_color1, mix_factor);    
    final_color *= sun_contribution;

    frag_color = vec4(final_color, 1.0);
}
