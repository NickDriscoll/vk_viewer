import structs;

ConstantBuffer<FrameUniforms> uniforms;

[[vk_binding(7, 0)]]
StructuredBuffer<ImguiVertex> imgui_vertices;

[[vk_binding(8, 0)]]
Sampler2D global_textures[];

[[vk::push_constant]]
cbuffer PushConstants
{
    uint font_idx;
    uint imgui_vertex_offset;
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
ImguiVertexOutput vertex_main(uint id : SV_VertexID) {
    ImguiVertex vert = imgui_vertices[id + imgui_vertex_offset];
    ImguiVertexOutput out;
    out.color = vert.color;
    out.uv = vert.uv;
    out.clip_out = mul(uniforms.clip_from_screen, float4(vert.position, 0.0, 1.0));

    return out;
}

[shader("fragment")]
float4 fragment_main(ImguiVertexOutput input) {
    return input.color * global_textures[font_idx].Sample(input.uv).r;
}