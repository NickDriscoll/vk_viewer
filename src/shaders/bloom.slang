import structs;
import bindings.frame_uniforms;
import bindings.global_textures;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint input_idx;
    uint output_idx;
    uint mip_level;
    uint phase_flag;
    uint hdr_index;
}

void downsample(uint3 thread_id) {
    Sampler2D input = global_textures[input_idx];
    RWTexture2D output = rw_images[output_idx];

    float2 input_dimensions;
    
    float _mips; // Unused
    input.GetDimensions(mip_level, input_dimensions.x, input_dimensions.y, _mips);
    // input.GetDimensions(mip_level + 1, output_dimensions.x, output_dimensions.y, _mips);
    float2 output_dimensions = floor(input_dimensions / 2.0);
    float2 texel_size = 1.0 / input_dimensions;
    float x = texel_size.x;
    float y = texel_size.y;

    // Early out for over-dispatched threads
    if (thread_id.x > (uint)output_dimensions.x || thread_id.y > (uint)output_dimensions.y) {
        return;
    }

    // Center of the sample points that will be used
    float2 sample_point = (thread_id.xy + 0.5) / output_dimensions;

    // Take 13 samples around current texel:
    // a - b - c
    // - j - k -
    // d - e - f
    // - l - m -
    // g - h - i
    // === ('e' is the current texel) ===
    float3 a = input.SampleLevel(float2(sample_point.x - 2 * x, sample_point.y + 2 * y), mip_level).rgb;
    float3 b = input.SampleLevel(float2(sample_point.x, sample_point.y + 2 * y), mip_level).rgb;
    float3 c = input.SampleLevel(float2(sample_point.x + 2 * x, sample_point.y + 2 * y), mip_level).rgb;

    float3 d = input.SampleLevel(float2(sample_point.x - 2 * x, sample_point.y), mip_level).rgb;
    float3 e = input.SampleLevel(float2(sample_point.x, sample_point.y), mip_level).rgb;
    float3 f = input.SampleLevel(float2(sample_point.x + 2 * x, sample_point.y), mip_level).rgb;

    float3 g = input.SampleLevel(float2(sample_point.x - 2 * x, sample_point.y - 2 * y), mip_level).rgb;
    float3 h = input.SampleLevel(float2(sample_point.x, sample_point.y - 2 * y), mip_level).rgb;
    float3 i = input.SampleLevel(float2(sample_point.x + 2 * x, sample_point.y - 2 * y), mip_level).rgb;

    float3 j = input.SampleLevel(float2(sample_point.x - x, sample_point.y + y), mip_level).rgb;
    float3 k = input.SampleLevel(float2(sample_point.x + x, sample_point.y + y), mip_level).rgb;
    float3 l = input.SampleLevel(float2(sample_point.x - x, sample_point.y - y), mip_level).rgb;
    float3 m = input.SampleLevel(float2(sample_point.x + x, sample_point.y - y), mip_level).rgb;

    // Apply weighted distribution:
    // 0.5 + 0.125 + 0.125 + 0.125 + 0.125 = 1
    // a,b,d,e * 0.125
    // b,c,e,f * 0.125
    // d,e,g,h * 0.125
    // e,f,h,i * 0.125
    // j,k,l,m * 0.5
    // This shows 5 square areas that are being sampled. But some of them overlap,
    // so to have an energy preserving downsample we need to make some adjustments.
    // The weights are the distributed, so that the sum of j,k,l,m (e.g.)
    // contribute 0.5 to the final color output. The code below is written
    // to effectively yield this sum. We get:
    // 0.125*5 + 0.03125*4 + 0.0625*4 = 1
    float3 downsample = e * 0.125;
    downsample += (a + c + g + i) * 0.03125;
    downsample += (b + d + f + h) * 0.0625;
    downsample += (j + k + l + m) * 0.125;
    downsample = max(downsample, 0.0001);

    output[thread_id.xy] = float4(downsample, 1.0);
}

void upsample(uint3 thread_id) {
    Sampler2D input = global_textures[input_idx];
    RWTexture2D output = rw_images[output_idx];

    float2 output_dimensions;
    float _mips; // Unused
    input.GetDimensions(mip_level - 1, output_dimensions.x, output_dimensions.y, _mips);
    float2 input_dimensions = floor(output_dimensions / 2.0);
    float2 texel_size = 1.0 / input_dimensions;

    // Early out for over-dispatched threads
    if (thread_id.x > (uint)output_dimensions.x || thread_id.y > (uint)output_dimensions.y) {
        return;
    }

    // The filter kernel is applied with a radius, specified in texture
    // coordinates, so that the radius will vary across mip resolutions.
    static const float filter_radius = 0.001;
    float x = filter_radius;
    float y = filter_radius;

    // Center of the sample points that will be used
    float2 sample_point = (thread_id.xy + 0.5) / output_dimensions;
    float3 output_sample;
    if (mip_level == 1) {
        output_sample = global_textures[hdr_index].Sample(sample_point).rgb;
    } else {
        output_sample = input.SampleLevel(sample_point, mip_level - 1).rgb;
    }

    // Take 9 samples around current texel:
    // a - b - c
    // d - e - f
    // g - h - i
    // === ('e' is the current texel) ===
    float3 a = input.SampleLevel(float2(sample_point.x - x, sample_point.y + y), mip_level).rgb;
    float3 b = input.SampleLevel(float2(sample_point.x, sample_point.y + y), mip_level).rgb;
    float3 c = input.SampleLevel(float2(sample_point.x + x, sample_point.y + y), mip_level).rgb;

    float3 d = input.SampleLevel(float2(sample_point.x - x, sample_point.y), mip_level).rgb;
    float3 e = input.SampleLevel(float2(sample_point.x, sample_point.y), mip_level).rgb;
    float3 f = input.SampleLevel(float2(sample_point.x + x, sample_point.y), mip_level).rgb;

    float3 g = input.SampleLevel(float2(sample_point.x - x, sample_point.y - y), mip_level).rgb;
    float3 h = input.SampleLevel(float2(sample_point.x, sample_point.y - y), mip_level).rgb;
    float3 i = input.SampleLevel(float2(sample_point.x + x, sample_point.y - y), mip_level).rgb;

    // Apply weighted distribution, by using a 3x3 tent filter:
    //  1   | 1 2 1 |
    // -- * | 2 4 2 |
    // 16   | 1 2 1 |
    float3 upsample = e * 4.0;
    upsample += (b + d + f + h) * 2.0;
    upsample += (a + c + g + i);
    upsample /= 16.0;

    output[thread_id.xy] = float4(output_sample + upsample, 1.0);
}

static const uint THREADS_X = 16;
static const uint THREADS_Y = 16;

[shader("compute")]
[numthreads(THREADS_X, THREADS_Y, 1)]
void compute_main(uint3 thread_id: SV_DispatchThreadID) {
    if (phase_flag != 0) {
        upsample(thread_id);
    } else {
        downsample(thread_id);
    }
}
