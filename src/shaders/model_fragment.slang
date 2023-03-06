import structs;
import lighting;
import routines;
import atmosphere;
import bindings.frame_uniforms;
import bindings.material_buffer;
import bindings.global_textures;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
}

[shader("fragment")]
float4 fragment_main(MeshVertexOutput input) : SV_Target0
{
    // Assemble input data
    float3 tangent = normalize(input.tangent);
    float3 bitangent = normalize(input.bitangent);
    float3 normal = normalize(input.normal);
    float3 view = normalize(uniforms.camera_position.xyz - input.position);
    float3x3 TBN = float3x3(tangent, bitangent, normal);
    Material my_mat = material_buffer[material_idx];

    // Discard the fragment if the alpha is below threshold
    float4 color_sample = my_mat.base_color * global_textures[my_mat.color_map_index].Sample(input.uv);
    if (color_sample.a < 0.1)
        discard;

    //Ambient Occlusion in R, Roughness in G, Metalness in B
    float4 metal_roughness_sample = global_textures[my_mat.metal_roughness_index].Sample(input.uv);
    float sampled_ao = metal_roughness_sample.r;
    float sampled_roughness = metal_roughness_sample.g;
    float sampled_metalness = metal_roughness_sample.b;

    // Get normal vector at this point
    float3 sampled_normal = 2.0 * global_textures[my_mat.normal_map_index].Sample(input.uv).xyz - 1.0;
    // float roughness = toksvigs_method(sampled_normal, my_mat.base_roughness * sampled_roughness);
    float roughness = my_mat.base_roughness * sampled_roughness;
    float metalness = my_mat.base_metalness * sampled_metalness;

    PBRSurface params;
    params.albedo = color_sample.rgb;
    params.shading_normal = normalize(mul(sampled_normal, TBN));
    params.geometry_normal = normal;
    params.view = view;
    params.roughness = roughness;
    params.metalness = metalness;

    // Outgoing radiance accumulator
    float3 final_radiance = float3(0.0);

    for (uint i = 0; i < uniforms.directional_light_count; i++) {
        DirectionalLight light = uniforms.directional_lights[i];
        float nighttime_factor = smoothstep(-0.1, 0.1, light.direction.z);

        ray_t ray = {
            float3(0.0, 0.0, earth_radius) + input.position,
            light.direction
        };
        float3 sky_sample = gather_atmosphere_irradiance(ray, light.direction.xyz, light.irradiance.rgb);

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
            global_array_textures[light.shadow_map_index],
            normal,
            light.direction.xyz,
            input.sunshadow_space_positions,
            uniforms.directional_lights[i].shadow_distances,
            input.clip_space_z,
            i
        );
        final_radiance += nighttime_factor * sun_shadowing.x * normalize(sky_sample) * directional_lighting(params, light);
    }

    //Emissive contribution
    final_radiance += 200.0 * global_textures[my_mat.emissive_index].Sample(input.uv).rgb;

    float3 ambient = float3(uniforms.ambient_factor) * params.albedo * sampled_ao;
    final_radiance += ambient;
    
    return float4(final_radiance, 1.0);
}
