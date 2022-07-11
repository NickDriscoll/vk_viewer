import structs;

// Bindings
ConstantBuffer<FrameUniforms> uniforms;

[[vk::binding(2, 0)]]
StructuredBuffer<Material> material_buffer;

[[vk_binding(4, 0)]]
Sampler2D global_textures[];

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
}

static const float PI = 3.14159265359;

// Everything is from https://learnopengl.com/PBR/Theory

float3 fresnel_schlick(float3 base_reflectivity, float3 halfway, float3 view)
{
    float quintic_term = clamp(1.0 - dot(halfway, view), 0.0, 1.0);
    return base_reflectivity +
           (float3(1.0) - base_reflectivity) * quintic_term * quintic_term * quintic_term * quintic_term * quintic_term;
}

// Trowbridge-Reitz
// Approximation of roughness at a point
float NDFggxtr(float3 normal, float3 halfway, float roughness)
{
    float thedot = max(dot(normal, halfway), 0.0);
    float r2 = roughness * roughness;
    float squared_term = thedot * thedot * (r2 - 1.0) + 1.0;
    return r2 / (PI * squared_term * squared_term);
}

float schlick_ggx(float3 normal, float3 view, float roughness)
{
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0; // Direct
    // float k = roughness * roughness / 2.0; //IBL
    float thedot = max(dot(normal, view), 0.0);
    return thedot / (thedot * (1.0 - k) + k);
}

float geometry_smith(float3 normal, float3 view, float3 light_direction, float roughness)
{
    float ggx1 = schlick_ggx(normal, view, roughness);
    float ggx2 = schlick_ggx(normal, light_direction, roughness);
    return ggx1 * ggx2;
}

float3 f_lambert(float3 color)
{
    return color / PI;
}

float3 pbr_mettalic_roughness(MeshVertexOutput input, float3x3 TBN, float3 normal, float3 view, Material my_mat)
{
    float roughness = 100.0;
    float metallic = 0.0;
    float ambient_occlusion = 1.0;

    // Discard the fragment if the alpha is below threshold
    float4 color_sample = global_textures[my_mat.color_map_index].Sample(input.uv);
    if (color_sample.a < 0.1)
        discard;
    float3 albedo = color_sample.rgb;

    // Get normal vector at this point
    float3 sampled_normal = 2.0 * global_textures[my_mat.normal_map_index].Sample(input.uv).xyz - 1.0;
    float3 world_normal = normalize(mul(sampled_normal, TBN));

    float3 halfway = normalize(view + uniforms.sun_direction.xyz);
    float3 ks = fresnel_schlick(float3(0.04), halfway, view);
    float3 kd = float3(1.0) - ks;
    kd *= 1.0 - metallic;

    // Cook-Torrence specular term evaluation
    float D = NDFggxtr(normal, halfway, roughness);
    float3 F = ks;
    float G = geometry_smith(normal, view, uniforms.sun_direction.xyz, roughness);
    float3 numerator = D * F * G;
    float denominator = 4.0 * max(dot(normal, view), 0.0) * max(dot(normal, uniforms.sun_direction.xyz), 0.0) + 0.0001;
    float3 cook_torrence = numerator / denominator;

    // Luminance from the sun
    float3 Lo = (kd * f_lambert(albedo) + cook_torrence) * uniforms.sun_radiance.xyz *
                max(0.0, dot(world_normal, uniforms.sun_direction.xyz));
    float sun_contribution = 1.0 - smoothstep(0.0, -0.05, uniforms.sun_direction.z);
    Lo *= sun_contribution;
    float3 ambient = float3(0.03) * albedo * ambient_occlusion;

    // Tone-mapping step
    float3 color = Lo + ambient;
    return color / (color + float3(1.0));
}