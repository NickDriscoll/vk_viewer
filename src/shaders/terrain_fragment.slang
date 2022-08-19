import structs;
import lighting;
import routines;
import fragment_bindings;
import bindings.global_textures;

float4 fragment_main(MeshVertexOutput input) : SV_Target0
{
    float3 tangent = normalize(input.tangent);
    float3 bitangent = normalize(input.bitangent);
    float3 normal = normalize(input.normal);
    float3 view = normalize(uniforms.camera_position.xyz - input.position);
    float3x3 TBN = float3x3(tangent, bitangent, normal);
    float2 uvs = input.uv;

    Material grass_material = material_buffer[material_idx];
    Material rock_material = material_buffer[material_idx + 1];

    float3 grass_albedo;
    float3 grass_normal;
    float grass_roughness;
    {
        float4 sample = global_textures[grass_material.color_map_index].Sample(uvs);
        grass_albedo = sample.rgb;
        grass_normal = normalize(2.0 * global_textures[grass_material.normal_map_index].Sample(uvs).xyz - 1.0);

        float4 metal_roughness_sample = global_textures[grass_material.metal_roughness_index].Sample(input.uv);
        grass_roughness = metal_roughness_sample.g;
    }

    float3 rock_albedo;
    float3 rock_normal;
    float rock_roughness;
    {
        float4 sample = global_textures[rock_material.color_map_index].Sample(uvs);
        rock_albedo = sample.rgb;
        rock_normal = normalize(2.0 * global_textures[rock_material.normal_map_index].Sample(uvs).xyz - 1.0);
        float4 metal_roughness_sample = global_textures[rock_material.metal_roughness_index].Sample(input.uv);
        rock_roughness = metal_roughness_sample.g;
    }

    float mix_factor = max(0.0, dot(normal, float3(0.0, 0.0, 1.0)));
    mix_factor = smoothstep(0.60, 0.70, mix_factor);

    float3 world_normal = mul(lerp(rock_normal, grass_normal, mix_factor), TBN);
    float roughness = lerp(grass_material.base_roughness * grass_roughness, rock_material.base_roughness * rock_roughness, mix_factor);

    //Compute sun color
    float sunzenith_dot = uniforms.sun_direction.z * 0.5 + 0.5;
    float3 sky_color = global_textures[uniforms.sunzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
    float3 halo_color = global_textures[uniforms.viewzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
    float3 sun_color = lerp(sky_color, halo_color, 0.2);

    PBRParams params;
    params.albedo = lerp(rock_albedo, grass_albedo, mix_factor);
    params.shading_normal = world_normal;
    params.geometry_normal = normal;
    params.view = view;
    params.light_direction = uniforms.sun_direction.xyz;
    params.light_radiance = lerp(uniforms.sun_radiance.rgb * halo_color, uniforms.sun_radiance.rgb, sunzenith_dot);
    params.roughness = roughness;

    float3 lit_color = pbr_mettalic_roughness(params);

    // Apply fog like GL_EXP2
    float3 final_color = exp2_fog(uniforms, input.position, lit_color, sun_color);

    return float4(final_color, 1.0);
}
