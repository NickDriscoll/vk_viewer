import structs;

[[vk_binding(3, 0)]]
StructuredBuffer<float4> vertex_positions;
[[vk_binding(3, 0)]]
StructuredBuffer<float4> vertex_tangents;
[[vk_binding(3, 0)]]
StructuredBuffer<float4> vertex_normals;
[[vk_binding(3, 0)]]
StructuredBuffer<float2> vertex_uvs;
