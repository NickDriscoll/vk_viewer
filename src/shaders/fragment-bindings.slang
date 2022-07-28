import structs;

ConstantBuffer<FrameUniforms> uniforms;

[[vk_binding(2, 0)]]
StructuredBuffer<Material> material_buffer;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
}