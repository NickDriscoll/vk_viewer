import structs;

[vk_binding(9, 0)]
Texture2D hdr_buffer;

[vk_binding(10, 0)]
RWStructuredBuffer<float3> histogram;

static const float EPSILON = 0.0005;
static const uint THREADS_X = 16;
static const uint THREADS_Y = 16;
static const uint GROUP_SIZE = THREADS_X * THREADS_Y;

// Taken from RealTime-Rendering vol 4 pg. 278
static const float3 RGB_TO_LUM = float3(0.2125, 0.7154, 0.0721);

shared uint histogram_shared[GROUP_SIZE];

// For a given color and luminance range, return the histogram bin index
uint colorToBin(float3 hdrColor, float minLogLum, float inverseLogLumRange) {
    // Convert our RGB value to Luminance, see note for RGB_TO_LUM macro above
    float lum = dot(hdrColor, RGB_TO_LUM);

    // Avoid taking the log of zero
    if (lum < EPSILON) {
        return 0;
    }

    // Calculate the log_2 luminance and express it as a value in [0.0, 1.0]
    // where 0.0 represents the minimum luminance, and 1.0 represents the max.
    float logLum = clamp((log2(lum) - minLogLum) * inverseLogLumRange, 0.0, 1.0);

    // Map [0, 1] to [1, 255]. The zeroth bin is handled by the epsilon check above.
    return uint(logLum * 254.0 + 1.0);
}

[shader("compute")]
[numthreads(THREADS_X, THREADS_Y, 1)]
void compute_main(uint3 thread_id : SV_DispatchThreadID, uint local_id : SV_GroupThreadID) {
    // Initialize the bin for this thread to 0
    //histogram_shared[local_id] = 0;
    GroupMemoryBarrier();

    //hdr_buffer.Load(float3());
}