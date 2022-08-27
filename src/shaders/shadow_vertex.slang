import structs;
import vertex_bindings;

[shader("vertex")]
ShadowVertexOutput vertex_main(ModelVertex vertex) {
    float4x4 model_matrix = instance_buffer[vertex.instance_id].world_from_model;

    ShadowVertexOutput out;
    out.clip_out = mul(uniforms.clip)
}