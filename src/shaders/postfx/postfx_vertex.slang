import structs;

static const float2 FULLSCREEN_TRIANGLE[] = {
    -1.0, -1.0,
    3.0, -1.0,
    -1.0, 3.0
};

[shader("vertex")]
float4 vertex_main(uint id: SV_VertexID) : SV_Position {
    float2 pos = FULLSCREEN_TRIANGLE[id];
    return float4(pos, 0.0, 1.0);
}