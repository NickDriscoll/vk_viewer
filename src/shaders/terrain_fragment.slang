import structs;
import lighting;

ConstantBuffer<FrameUniforms> uniforms;

[[vk_binding(2, 0)]]
StructuredBuffer<Material> material_buffer;

[[vk_binding(4, 0)]]
Sampler2D global_textures[];

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
}

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
    params.surface_normal = world_normal;
    params.geometry_normal = normal;
    params.view = view;

    return float4(pbr_mettalic_roughness(input, params, uniforms), 1.0);
}
