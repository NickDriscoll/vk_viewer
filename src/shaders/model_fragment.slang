import structs;
import lighting;
import fragment_bindings;

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
    float4 color_sample = global_textures[my_mat.color_map_index].Sample(input.uv);
    if (color_sample.a < 0.1)
        discard;
    params.albedo = color_sample.rgb;

    // Get normal vector at this point
    float3 sampled_normal = 2.0 * global_textures[my_mat.normal_map_index].Sample(input.uv).xyz - 1.0;
    params.surface_normal = normalize(mul(sampled_normal, TBN));
    params.geometry_normal = normal;
    params.view = view;
    params.light_direction = uniforms.sun_direction.xyz;
    params.light_radiance = uniforms.sun_radiance.rgb;

    return float4(pbr_mettalic_roughness(params), 1.0);
}
