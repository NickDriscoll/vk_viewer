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

const float PI = 3.14159265359;

//Everything is from https://learnopengl.com/PBR/Theory

vec3 fresnel_schlick(vec3 base_reflectivity, vec3 halfway, vec3 view) {
    float quintic_term = clamp(1.0 - dot(halfway, view), 0.0, 1.0);
    return base_reflectivity + (vec3(1.0) - base_reflectivity) * quintic_term * quintic_term * quintic_term * quintic_term * quintic_term;
}

//Trowbridge-Reitz
//Approximation of roughness at a point
float NDFggxtr(vec3 normal, vec3 halfway, float roughness) {
    float thedot = max(dot(normal, halfway), 0.0);
    float r2 = roughness * roughness;
    float squared_term = thedot*thedot * (r2 - 1.0) + 1.0;
    return r2 / (PI * squared_term * squared_term);
}

float schlick_ggx(vec3 normal, vec3 view, float roughness) {
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0; //Direct
    //float k = roughness * roughness / 2.0; //IBL
    float thedot = max(dot(normal, view), 0.0);
    return thedot / (thedot * (1.0 - k) + k);
}

float geometry_smith(vec3 normal, vec3 view, vec3 light_direction, float roughness) {
    float ggx1 = schlick_ggx(normal, view, roughness);
    float ggx2 = schlick_ggx(normal, light_direction, roughness);
    return ggx1 * ggx2;
}

vec3 f_lambert(vec3 color) {
    return color / PI;
}

void main() {
    //Assemble input data
    vec3 tangent = normalize(f_tangent);
    vec3 bitangent = normalize(f_bitangent);
    vec3 normal = normalize(f_normal);
    vec3 view = normalize(camera_position.xyz - f_position);
    mat3 TBN = mat3(tangent, bitangent, normal);
    Material my_mat = global_materials[material_idx];

    float roughness = 0.3;
    float metallic = 0.0;
    float ambient_occlusion = 1.0;

    //Discard the fragment if the alpha is below threshold
    vec4 color_sample = texture(global_textures[my_mat.color_map_index], f_uv);
    if (color_sample.a < 0.1) discard;
    vec3 albedo = color_sample.rgb;

    //Get normal vector at this point
    vec3 sampled_normal = 2.0 * texture(global_textures[my_mat.normal_map_index], f_uv).xyz - 1.0;
    vec3 world_normal = normalize(TBN * sampled_normal);
    
    vec3 halfway = normalize(view + sun_direction.xyz);
    vec3 ks = fresnel_schlick(vec3(0.04), halfway, view);
    vec3 kd = vec3(1.0) - ks;
    kd *= 1.0 - metallic;

    //Cook-Torrence specular term evaluation
    float D = NDFggxtr(normal, halfway, roughness);
    vec3 F = ks;
    float G = geometry_smith(normal, view, sun_direction.xyz, roughness);
    vec3 numerator = D * F * G;
    float denominator = 4.0 * max(dot(normal, view), 0.0) * max(dot(normal, sun_direction.xyz), 0.0) + 0.0001;
    vec3 cook_torrence = numerator / denominator;

    //Luminance from the sun
    vec3 Lo = (kd * f_lambert(albedo) + cook_torrence) * sun_radiance.xyz * max(0.0, dot(world_normal, sun_direction.xyz));
    vec3 ambient = vec3(0.03) * albedo * ambient_occlusion;
    
    vec3 color = Lo + ambient;
    color = color / (color + vec3(1.0));

    frag_color = vec4(color, 1.0);
}
