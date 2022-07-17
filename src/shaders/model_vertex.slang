import structs;
import vertex_bindings;

[shader("vertex")]
MeshVertexOutput vertex_main(ModelVertex vertex) {
    float4x4 model_matrix = instance_buffer[vertex.instance_id].world_from_model;
    float4x4 normal_matrix = instance_buffer[vertex.instance_id].normal_matrix;

    float3 position = vertex_positions[vertex.id + position_buffer_offset].xyz;
    float3 world_position = mul(model_matrix, float4(lerp(vertex.position, position, uniforms.prog_inter), 1.0)).xyz;

    float3 tangent = vertex_tangents[vertex.id + tangent_buffer_offset].xyz;
    float3 world_tangent = normalize(mul(normal_matrix, float4(lerp(vertex.tangent, tangent, uniforms.prog_inter), 1.0)).xyz);

    float3 bitangent = vertex_bitangents[vertex.id + bitangent_buffer_offset].xyz;
    float3 world_bitangent = normalize(mul(normal_matrix, float4(lerp(vertex.bitangent, bitangent, uniforms.prog_inter), 1.0)).xyz);

    float3 normal = cross(tangent, bitangent);
    float3 world_normal = normalize(mul(normal_matrix, float4(lerp(vertex.normal, normal, uniforms.prog_inter), 1.0)).xyz);

    float2 uv = vertex_uvs[vertex.id + uv_buffer_offset];

    //Assemble and return output
    MeshVertexOutput output;
    output.clip_out = mul(uniforms.clip_from_world, float4(world_position, 1.0));
    output.position = world_position;
    output.normal = world_normal;
    output.uv = lerp(vertex.uv, uv, uniforms.prog_inter);
    output.tangent = world_tangent;
    output.bitangent = world_bitangent;

    return output;
}
