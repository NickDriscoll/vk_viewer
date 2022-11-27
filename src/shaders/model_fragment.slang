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

    float4 metal_roughness_sample = global_textures[my_mat.metal_roughness_index].Sample(input.uv);
    float sampled_ao = metal_roughness_sample.r;
    float sampled_roughness = metal_roughness_sample.g;
    float sampled_metalness = metal_roughness_sample.b;

    // Compute sun color
    float sunzenith_dot = uniforms.sun_direction.z * 0.5 + 0.5;
    float3 halo_color = float3(1.0);
    float3 sky_color = float3(1.0);
    float3 sun_color;
    if (!uniforms.real_sky) {
        sky_color = global_textures[uniforms.sunzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
        halo_color = global_textures[uniforms.viewzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
    }
    sun_color = lerp(sky_color, halo_color, 0.2);

    // Get normal vector at this point
    float3 sampled_normal = 2.0 * global_textures[my_mat.normal_map_index].Sample(input.uv).xyz - 1.0;
    float roughness = toksvigs_method(sampled_normal, my_mat.base_roughness * sampled_roughness);

    PBRSurfaceParams params;
    params.albedo = color_sample.rgb;
    params.shading_normal = normalize(mul(sampled_normal, TBN));
    params.geometry_normal = normal;
    params.view = view;
    params.roughness = roughness;
    params.metalness = sampled_metalness;

    DirectionalLight sunlight = {
        uniforms.sun_direction.xyz,
        uniforms.sun_intensity * lerp(uniforms.sun_radiance.rgb, uniforms.sun_radiance.rgb, sunzenith_dot)
    };

    float3 final_color = float3(0.0);

    float2 sun_shadowing = compute_cascade_shadowing(
        global_array_textures[uniforms.sun_shadowmap_idx],
        normal,
        uniforms.sun_direction.xyz,
        input.sunshadow_space_positions,
        uniforms.sun_shadow_distances,
        input.clip_space_z
    );
    final_color = sun_shadowing.x * pbr_mettalic_roughness(params, sunlight);
    final_color += global_textures[my_mat.emissive_index].Sample(input.uv).rgb;

    if (!uniforms.real_sky) {
        // Apply fog like GL_EXP2
        final_color = exp2_fog(uniforms, input.position, final_color, sun_color);
    }

    // float3 ambient = float3(uniforms.ambient_factor) * params.albedo * sampled_ao;
    float3 ambient = float3(uniforms.ambient_factor) * params.albedo;
    final_color = final_color + ambient;
    
    return float4(final_color, 1.0);
}
