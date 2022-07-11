import structs;

ConstantBuffer<FrameUniforms> uniforms;
StructuredBuffer<InstanceData> instance_buffer;

[[vk_binding(3, 0)]]
StructuredBuffer<float4> vertex_positions;
StructuredBuffer<float3> vertex_tangents;
StructuredBuffer<float3> vertex_bitangents;
StructuredBuffer<float2> vertex_uvs;