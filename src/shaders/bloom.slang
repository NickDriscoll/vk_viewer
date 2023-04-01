import structs;
import bindings.global_textures;

static const float EPSILON = 0.0005;
static const uint THREADS_X = 16;
static const uint THREADS_Y = 16;
static const uint GROUP_SIZE = THREADS_X * THREADS_Y;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint input_idx;
    uint output_idx;
    uint mip_level;
}

[shader("compute")]
[numthreads(THREADS_X, THREADS_Y, 1)]
void compute_main(uint3 thread_id: SV_DispatchThreadID, uint local_id: SV_GroupThreadID) {
    Sampler2D input = global_textures[input_idx];
    RWTexture2D output = global_rw_textures[output_idx];

    output[thread_id.xy] = float4(1.0, 0.0, 0.0, 1.0);
}