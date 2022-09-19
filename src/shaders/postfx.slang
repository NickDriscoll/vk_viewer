import structs;
import bindings.frame_uniforms;
import bindings.global_textures;

static const float2 FULLSCREEN_TRIANGLE[] = {
    -1.0, 1.0,
    3.0, 1.0,
    -1.0, -3.0
};

[[vk::push_constant]]
cbuffer PushConstants
{
    uint hdr_buffer_index;
}

struct PostFXVertexOutput {
    float4 clip_pos : SV_Position;
    float2 uv;
}

[shader("vertex")]
PostFXVertexOutput vertex_main(uint id: SV_VertexID) {
    PostFXVertexOutput out;
    out.uv = FULLSCREEN_TRIANGLE[id];
    out.clip_pos = float4(out.uv, 0.0, 1.0);
    return out;
}

[shader("fragment")]
float4 fragment_main(PostFXVertexOutput input) {
    float2 uv = (input.uv + 1.0) / 2.0;
    Sampler2D hdr_buffer = global_textures[hdr_buffer_index];

    float3 final_color = hdr_buffer.Sample(uv).xyz;
    final_color *= uniforms.exposure;
    //final_color = (final_color.r + final_color.g + final_color.b) / 3.0;

    //Tonemapping step

    // Narkowicz ACES
    final_color = (final_color * (2.51 * final_color + 0.03)) / (final_color * (2.43 * final_color + 0.59) + 0.14);
    final_color = saturate(final_color);
    //final_color = final_color / (final_color + float3(1.0));

    return float4(final_color, 1.0);
}