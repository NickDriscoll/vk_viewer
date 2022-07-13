import structs;

ConstantBuffer<FrameUniforms> uniforms;
StructuredBuffer<InstanceData> instance_buffer;

[[vk_binding(3, 0)]]
StructuredBuffer<float4> vertex_positions;
[[vk_binding(4, 0)]]
StructuredBuffer<float4> vertex_tangents;
[[vk_binding(5, 0)]]
StructuredBuffer<float4> vertex_bitangents;
[[vk_binding(6, 0)]]
StructuredBuffer<float2> vertex_uvs;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
    uint position_buffer_offset;
    uint tangent_buffer_offset;
    uint bitangent_buffer_offset;
    uint uv_buffer_offset;
}
