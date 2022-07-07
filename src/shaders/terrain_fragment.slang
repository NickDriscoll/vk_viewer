import structs;
import lighting;

ConstantBuffer<FrameUniforms> uniforms;

[[vk_binding(3, 0)]]
StructuredBuffer<Material> material_buffer;

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

    Material grass_material = global_materials[first_material_idx];
    vec3 grass_color = texture(global_textures[grass_material.color_map_index], uvs).rgb;
    vec3 grass_normal = normalize(2.0 * texture(global_textures[grass_material.normal_map_index], uvs).xyz - 1.0);

    Material rock_material = global_materials[first_material_idx + 1];
    vec3 rock_color = texture(global_textures[rock_material.color_map_index], uvs).rgb;
    vec3 rock_normal = normalize(2.0 * texture(global_textures[rock_material.normal_map_index], uvs).xyz - 1.0);

    float mix_factor = max(0.0, dot(normal, vec3(0.0, 0.0, 1.0)));
    mix_factor = smoothstep(0.60, 0.70, mix_factor);

    vec3 world_normal = TBN * mix(rock_normal, grass_normal, mix_factor);
    float sun_contribution = max(0.0, dot(world_normal, sun_direction.xyz) + 0.1);
    sun_contribution *= 1.0 - smoothstep(0.0, -0.05, sun_direction.z);

    vec3 final_color = mix(rock_color, grass_color, mix_factor);
    final_color *= max(LIGHTING_MIN, sun_contribution);

    return float4(pbr_mettalic_roughness(input, TBN, normal, view, my_mat, uniforms), 1.0);
}
