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

    PBRSurface params;
    params.albedo = lerp(rock_info.albedo, grass_info.albedo, mix_factor);
    params.shading_normal = world_normal;
    params.geometry_normal = normal;
    params.view = view;
    params.roughness = roughness;
    params.metalness = metalness;

    // Outgoing radiance accumulator
    float3 final_radiance = float3(0.0);

    for (uint i = 0; i < uniforms.directional_light_count; i++) {
        uint light_idx = i;
        DirectionalLight light = uniforms.directional_lights[i];

        // Compute sun color
        float sunzenith_dot = light.direction.z * 0.5 + 0.5;
        float3 halo_color = float3(1.0);
        float3 sky_color = float3(1.0);
        float3 sun_color;
        if (!uniforms.real_sky) {
            sky_color = global_textures[uniforms.sunzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
            halo_color = global_textures[uniforms.viewzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
        }
        sun_color = lerp(sky_color, halo_color, 0.2);

        float2 sun_shadowing = compute_cascade_shadowing(
            global_array_textures[uniforms.sun_shadowmap_idx],
            normal,
            light.direction.xyz,
            input.sunshadow_space_positions,
            uniforms.sun_shadow_distances,
            input.clip_space_z,
            0
        );
        final_radiance += sun_shadowing.x * directional_lighting(params, light);

        if (!uniforms.real_sky) {
            // Apply fog like GL_EXP2
            final_radiance = exp2_fog(uniforms, input.position, final_radiance, sun_color);
        }
    }

    float3 ambient = float3(uniforms.ambient_factor) * params.albedo * ao;
    //float3 ambient = float3(uniforms.ambient_factor) * params.albedo;
    final_radiance += ambient;

    //Determine which cascade the fragment is in
    float3 adj_shadow_space_pos;
    int shadow_cascade = -1;
    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        uint cascade_index = i + (SHADOW_CASCADE_COUNT * 0);
        float cascade_distance = uniforms.sun_shadow_distances[(cascade_index + 1) / 4][(cascade_index + 1) % 4];
        if (input.clip_space_z < cascade_distance) {
            adj_shadow_space_pos = float3(float2(input.sunshadow_space_positions[i].xy * 0.5 + 0.5), input.sunshadow_space_positions[i].z);
            if (!(
                adj_shadow_space_pos.z < 0.0 ||
                adj_shadow_space_pos.z > 1.0 ||
                adj_shadow_space_pos.x < 0.0 ||
                adj_shadow_space_pos.x > 1.0 ||
                adj_shadow_space_pos.y < 0.0 ||
                adj_shadow_space_pos.y > 1.0
            )) {
                shadow_cascade = i;
            }
            break;
        }
    }
    float2 f_shadow_pos = adj_shadow_space_pos.xy;
    float3 sample_uv = float3(f_shadow_pos.xy, (float)shadow_cascade);

    //return float4(final_radiance, 1.0);
    return float4(global_array_textures[uniforms.sun_shadowmap_idx].Sample(float3(uvs, 1.0)).r);
}
