import structs;
import vertex_bindings;

[shader("vertex")]
MeshVertexOutput vertex_main(ModelVertex vertex) {
    float4x4 model_matrix = instance_buffer[vertex.instance_id].world_from_model;
    float4x4 normal_matrix = instance_buffer[vertex.instance_id].normal_matrix;

    float3 world_position = mul(model_matrix, float4(vertex.position, 1.0)).xyz;
    float3 position = vertex_positions[vertex.id].xyz;
    float3 n_world_position = mul(model_matrix, float4(position, 1.0)).xyz;

    float3 world_tangent = normalize(mul(normal_matrix, float4(vertex.tangent, 1.0)).xyz);
    float3 world_bitangent = normalize(mul(normal_matrix, float4(vertex.bitangent, 1.0)).xyz);
    float3 world_normal = normalize(mul(normal_matrix, float4(vertex.normal, 1.0)).xyz);

    //Assemble and return output
    MeshVertexOutput output;
    output.clip_out = mul(uniforms.clip_from_world, float4(lerp(world_position, n_world_position, uniforms.pos_inter), 1.0));
    output.position = world_position;
    output.normal = world_normal;
    output.uv = vertex.uv;
    output.tangent = world_tangent;
    output.bitangent = world_bitangent;

    return output;
}
