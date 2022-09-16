import structs;
import bindings.frame_uniforms;
import bindings.instance_buffer;
import vertex_bindings;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
    uint position_buffer_offset;
    uint tangent_buffer_offset;
    uint normal_buffer_offset;
    uint uv_buffer_offset;
}

[shader("vertex")]
MeshVertexOutput vertex_main(ModelVertex vertex) {
    float4x4 model_matrix = instance_buffer[vertex.instance_id].world_from_model;
    float4x4 normal_matrix = instance_buffer[vertex.instance_id].normal_matrix;

    float3 position = vertex_positions[vertex.id + position_buffer_offset].xyz;
    float3 world_position = mul(model_matrix, float4(position, 1.0)).xyz;

    float4 tan = vertex_tangents[vertex.id + tangent_buffer_offset];
    float bitangent_sign = tan.a;
    float3 tangent = tan.xyz;
    float3 world_tangent = normalize(mul(normal_matrix, float4(tangent, 1.0)).xyz);

    float3 normal = vertex_normals[vertex.id + normal_buffer_offset].xyz;
    float3 world_normal = normalize(mul(normal_matrix, float4(normal, 1.0)).xyz);

    float3 bitangent = bitangent_sign * cross(normal, tangent);
    float3 world_bitangent = normalize(mul(normal_matrix, float4(bitangent, 1.0)).xyz);

    float2 uv = vertex_uvs[vertex.id + uv_buffer_offset];

    //Assemble and return output
    MeshVertexOutput output;
    output.clip_out = mul(uniforms.clip_from_world, float4(world_position, 1.0));
    output.position = world_position;
    output.normal = world_normal;
    output.uv = uv;
    output.tangent = world_tangent;
    output.bitangent = world_bitangent;

    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        output.sunshadow_space_positions[i] = mul(uniforms.sun_shadow_matrices[i], float4(world_position, 1.0));
    }

    output.clip_space_z = output.clip_out.z;
    return output;
}
