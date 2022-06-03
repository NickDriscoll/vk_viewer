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

const float PI = 3.141592654;

//Everything is from https://learnopengl.com/PBR/Theory

vec3 fresnel_schlick(vec3 base_reflectivity, vec3 halfway, vec3 view) {
    float quintic_term = (1.0 - dot(halfway, view));
    return base_reflectivity +
           (vec3(1.0) - base_reflectivity) * quintic_term * quintic_term * quintic_term * quintic_term * quintic_term;
}

//Approximation of roughness at a point
float NDFggxtr(vec3 normal, vec3 halfway, float roughness) {
    float thedot = dot(normal, halfway);
    float squared_term = thedot*thedot * (roughness*roughness - 1.0) + 1.0;
    return roughness * roughness / (PI * squared_term * squared_term);
}

float schlick_ggx(vec3 normal, vec3 view, float roughness) {
    float thedot = dot(normal, view);
    return thedot / (thedot * (1.0 - roughness) + roughness);
}

vec3 f_lambert(vec3 color) {
    return color * PI;
}

vec3 f_cook_torrence() {

}

void main() {
    //Assemble input data
    vec3 tangent = normalize(f_tangent);
    vec3 bitangent = normalize(f_bitangent);
    vec3 normal = normalize(f_normal);
    mat3 TBN = mat3(tangent, bitangent, normal);
    Material my_mat = global_materials[material_idx];

    //Discard the fragment if the alpha is below threshold
    vec4 color_sample = texture(global_textures[my_mat.color_map_index], f_uv);
    if (color_sample.a < 0.1) discard;
    vec3 base_color = color_sample.rgb;

    vec3 sampled_normal = 2.0 * texture(global_textures[my_mat.normal_map_index], f_uv).xyz - 1.0;
    vec3 world_normal = TBN * sampled_normal;
    float sun_contribution = max(0.0, dot(world_normal, sun_direction));
    sun_contribution *= 1.0 - smoothstep(0.0, -0.05, sun_direction.z);

    vec3 final_color = base_color;
    final_color *= max(LIGHTING_MIN, sun_contribution);

    frag_color = vec4(final_color, 1.0);
}
