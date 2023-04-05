import structs;
import bindings.global_textures;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint input_idx;
    uint output_idx;
    uint mip_level;
}

static const uint THREADS_X = 16;
static const uint THREADS_Y = 16;

[shader("compute")]
[numthreads(THREADS_X, THREADS_Y, 1)]
void compute_main(uint3 thread_id: SV_DispatchThreadID, uint local_id: SV_GroupThreadID) {
    Sampler2D input = global_textures[input_idx];
    RWTexture2D output = rw_images[output_idx];

    float2 input_dimensions;
    float mips;
    input.GetDimensions(mip_level, input_dimensions.x, input_dimensions.y, mips);
    float2 texel_size = 1.0 / input_dimensions;
    float2 sample_point = (thread_id.xy + 0.5) * texel_size;

    output[thread_id.xy] = input.SampleLevel(sample_point, (float)mip_level) * float4(1.0, 0.5, 0.25, 1.0);
}
