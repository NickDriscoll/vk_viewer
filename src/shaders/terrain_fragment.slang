import structs;
import lighting;
import routines;
import bindings.frame_uniforms;
import bindings.material_buffer;
import bindings.global_textures;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
}

struct MaterialInfo {
    float3 albedo;
    float3 normal;
    float roughness;
    float metalness;
    float ao;
}

MaterialInfo info_from_material(Material mat, float2 uvs) {
    MaterialInfo out;

    float4 sample = global_textures[mat.color_map_index].Sample(uvs);
    out.albedo = sample.rgb;
    float3 n_sample = 2.0 * global_textures[mat.normal_map_index].Sample(uvs).xyz - 1.0;
    out.normal = normalize(n_sample);

    float4 metal_roughness_sample = global_textures[mat.metal_roughness_index].Sample(uvs);

    // out.roughness = toksvigs_method(n_sample, mat.base_roughness * metal_roughness_sample.g);
    out.roughness = mat.base_roughness * metal_roughness_sample.g;
    out.metalness = mat.base_metalness * metal_roughness_sample.b;
    out.ao = metal_roughness_sample.r;

    return out;
}

[shader("fragment")]
float4 fragment_main(MeshVertexOutput input) : SV_Target0
{
    float3 tangent = normalize(input.tangent);
    float3 bitangent = normalize(input.bitangent);
    float3 normal = normalize(input.normal);
    float3 view = normalize(uniforms.camera_position.xyz - input.position);
    float3x3 TBN = float3x3(tangent, bitangent, normal);
    float2 uvs = input.uv;

    Material grass_material = material_buffer[material_idx];
    MaterialInfo grass_info = info_from_material(grass_material, input.uv);

    Material rock_material = material_buffer[material_idx + 1];
    MaterialInfo rock_info = info_from_material(rock_material, input.uv);

    float mix_factor = max(0.0, dot(normal, float3(0.0, 0.0, 1.0)));
    mix_factor = smoothstep(0.60, 0.70, mix_factor);

    float3 world_normal = mul(lerp(rock_info.normal, grass_info.normal, mix_factor), TBN);
    float roughness = lerp(rock_info.roughness, grass_info.roughness, mix_factor);
    float metalness = lerp(rock_info.metalness, grass_info.metalness, mix_factor);
    float ao = lerp(rock_info.ao, grass_info.ao, mix_factor);

    // Compute sun color
    float sunzenith_dot = uniforms.sun_direction.z * 0.5 + 0.5;
    float3 halo_color = float3(1.0);
    float3 sky_color = float3(1.0);
    float3 sun_color;
    if (!uniforms.real_sky) {
        float3 sky_color = global_textures[uniforms.sunzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
        halo_color = global_textures[uniforms.viewzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
    }
    sun_color = lerp(sky_color, halo_color, 0.2);

    PBRSurface params;
    params.albedo = lerp(rock_info.albedo, grass_info.albedo, mix_factor);
    params.shading_normal = world_normal;
    params.geometry_normal = normal;
    params.view = view;
    params.roughness = roughness;
    params.metalness = metalness;

    DirectionalLight sunlight = {
        uniforms.sun_direction.xyz,
        uniforms.sun_irradiance.rgb
    };

    float3 final_radiance = float3(0.0);

    float2 sun_shadowing = compute_cascade_shadowing(
        global_array_textures[uniforms.sun_shadowmap_idx],
        normal,
        uniforms.sun_direction.xyz,
        input.sunshadow_space_positions,
        uniforms.sun_shadow_distances,
        input.clip_space_z
    );
    final_radiance = sun_shadowing.x * directional_lighting(params, sunlight);

    if (!uniforms.real_sky) {
        // Apply fog like GL_EXP2
        final_radiance = exp2_fog(uniforms, input.position, final_radiance, sun_color);
    }

    //float3 ambient = float3(uniforms.ambient_factor) * params.albedo * ao;
    float3 ambient = float3(uniforms.ambient_factor) * params.albedo;
    final_radiance = final_radiance + ambient;

    return float4(final_radiance, 1.0);
}
