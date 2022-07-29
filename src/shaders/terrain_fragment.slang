import structs;
import lighting;
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
    {
        float4 sample = global_textures[grass_material.color_map_index].Sample(uvs);
        if (sample.a < 0.1)
            discard;
        grass_albedo = sample.rgb;
        grass_normal = normalize(2.0 * global_textures[grass_material.normal_map_index].Sample(uvs).xyz - 1.0);
    }

    float3 rock_albedo;
    float3 rock_normal;
    {
        float4 sample = global_textures[rock_material.color_map_index].Sample(uvs);
        if (sample.a < 0.1)
            discard;
        rock_albedo = sample.rgb;
        rock_normal = normalize(2.0 * global_textures[rock_material.normal_map_index].Sample(uvs).xyz - 1.0);
    }

    float mix_factor = max(0.0, dot(normal, float3(0.0, 0.0, 1.0)));
    mix_factor = smoothstep(0.60, 0.70, mix_factor);
    
    float3 world_normal = mul(lerp(rock_normal, grass_normal, mix_factor), TBN);

    PBRParams params;
    params.albedo = lerp(rock_albedo, grass_albedo, mix_factor);
    params.shading_normal = world_normal;
    params.geometry_normal = normal;
    params.view = view;
    params.light_direction = uniforms.sun_direction.xyz;
    params.light_radiance = uniforms.sun_radiance.rgb;

    float3 lit_color = pbr_mettalic_roughness(params);

    // Apply fog like GL_EXP2
    float real_depth = length(uniforms.camera_position.xyz - input.position);
    float base = 0.001 * uniforms.fog_density * real_depth;
    float fog_factor = exp(-(base * base));

    float3 final_color = lerp(float3(1.0), lit_color, fog_factor);
    return float4(final_color, 1.0);
}
