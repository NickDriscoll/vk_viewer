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
    uint bloom_buffer_index;
}

struct PostFXVertexOutput {
    float4 clip_pos : SV_Position;
    float2 uv;
}

[shader("vertex")]
PostFXVertexOutput vertex_main(uint id: SV_VertexID) {
    PostFXVertexOutput out;
    out.uv = (FULLSCREEN_TRIANGLE[id] + 1.0) / 2.0;
    out.clip_pos = float4(FULLSCREEN_TRIANGLE[id], 0.0, 1.0);
    return out;
}

[shader("fragment")]
float4 fragment_main(PostFXVertexOutput input) {
    Sampler2D hdr_buffer = global_textures[hdr_buffer_index];
    Sampler2D bloom_buffer = global_textures[bloom_buffer_index];

    float3 final_color = hdr_buffer.Sample(input.uv).rgb;
    final_color += bloom_buffer.Sample(input.uv).rgb * uniforms.bloom_strength;
    final_color *= uniforms.exposure;

    // Tonemapping step
    //final_color = final_color / (final_color + float3(1.0));
    //final_color = pow(final_color, float3(1.0 / 2.2));

    // Narkowicz ACES
    final_color = (final_color * (2.51 * final_color + 0.03)) / (final_color * (2.43 * final_color + 0.59) + 0.14);
    final_color = saturate(final_color);
    
    return float4(final_color, 1.0);
}