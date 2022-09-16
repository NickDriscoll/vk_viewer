import structs;
import bindings.frame_uniforms;
import bindings.imgui_vertices;
import bindings.global_textures;


[[vk::push_constant]]
cbuffer PushConstants
{
    uint font_idx;
    uint imgui_vertex_offset;
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