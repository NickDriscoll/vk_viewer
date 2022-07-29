import structs;
import lighting;
import fragment_bindings;
import bindings.global_textures;

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
    PBRParams params;

    // Discard the fragment if the alpha is below threshold
    float4 color_sample = my_mat.base_color * global_textures[my_mat.color_map_index].Sample(input.uv);
    if (color_sample.a < 0.1)
        discard;
    params.albedo = color_sample.rgb;

    // Get normal vector at this point
    float3 sampled_normal = 2.0 * global_textures[my_mat.normal_map_index].Sample(input.uv).xyz - 1.0;
    params.shading_normal = normalize(mul(sampled_normal, TBN));
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
