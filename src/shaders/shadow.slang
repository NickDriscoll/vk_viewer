import structs;
import bindings.frame_uniforms;
import bindings.instance_buffer;
import vertex_bindings;
import bindings.global_textures;
import bindings.material_buffer;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
    uint position_buffer_offset;
    uint uv_buffer_offset;
}

[shader("vertex")]
ShadowVertexOutput vertex_main(ModelVertex vertex) {
    float4x4 model_matrix = instance_buffer[vertex.instance_id].world_from_model;
    float3 position = vertex_positions[vertex.id + position_buffer_offset].xyz;
    float3 world_position = mul(model_matrix, float4(position, 1.0)).xyz;

    ShadowVertexOutput out;
    out.clip_out = mul(uniforms.sun_shadow_matrices[0], float4(world_position, 1.0));
    out.uv = vertex_uvs[vertex.id + uv_buffer_offset];

    return out;
}

[shader("fragment")]
void fragment_main(ShadowVertexOutput input) {
    Material my_mat = material_buffer[material_idx];

    if (global_textures[my_mat.color_map_index].Sample(input.uv).a < 0.1) {
        discard;
    }
}
