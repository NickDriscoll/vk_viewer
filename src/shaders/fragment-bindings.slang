import structs;

ConstantBuffer<FrameUniforms> uniforms;

[[vk_binding(2, 0)]]
StructuredBuffer<Material> material_buffer;

[[vk_binding(4, 0)]]
Sampler2D global_textures[];

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
}