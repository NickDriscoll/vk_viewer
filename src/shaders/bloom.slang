import structs;
import bindings.global_textures;

static const float EPSILON = 0.0005;
static const uint THREADS_X = 16;
static const uint THREADS_Y = 16;
static const uint GROUP_SIZE = THREADS_X * THREADS_Y;

[shader("compute")]
[numthreads(THREADS_X, THREADS_Y, 1)]
void compute_main(uint3 thread_id : SV_DispatchThreadID, uint local_id : SV_GroupThreadID) {
    
}