import structs;
import vertex_bindings;

[[vk_binding(4, 0)]]
Sampler2D global_textures[];

[[vk::push_constant]]
cbuffer PushConstants
{
    uint font_idx;
}

struct ImguiVertex {
    float2 position : POSITION;
    float2 uv : UVS;
    float4 color : COLOR;
}

struct ImguiVertexOutput {
    float4 clip_out: SV_Position;
    float2 uv;
    float4 color;
}

[shader("vertex")]
ImguiVertexOutput vertex_main(ImguiVertex input) {
    ImguiVertexOutput out;
    out.color = input.color;
    out.uv = input.uv;
    out.clip_out = mul(uniforms.clip_from_screen, float4(input.position, 0.0, 1.0));

    return out;
}

[shader("fragment")]
float4 fragment_main(ImguiVertexOutput input) {
    return input.color * global_textures[font_idx].Sample(input.uv).r;
}