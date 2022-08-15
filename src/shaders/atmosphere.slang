import structs;
import bindings.global_textures;
import routines;

// Bindings
ConstantBuffer<FrameUniforms> uniforms;

static const float3 SKYBOX_NONINDEXED_POSITIONS[] = {
    // Front
    -1.0, -1.0, -1.0,
    1.0, -1.0, -1.0,
    -1.0, 1.0, -1.0,
    1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0,
    1.0, -1.0, -1.0,

    // Left
    -1.0, -1.0, -1.0,
    -1.0, 1.0, -1.0,
    -1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0,
    -1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0,

    // Right
    1.0, 1.0, -1.0,
    1.0, -1.0, -1.0,
    1.0, -1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, -1.0,
    1.0, -1.0, 1.0,

    // Back
    -1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, -1.0, 1.0,
    -1.0, -1.0, 1.0,

    // Bottom
    -1.0, -1.0, 1.0,
    1.0, -1.0, -1.0,
    -1.0, -1.0, -1.0,
    -1.0, -1.0, 1.0,
    1.0, -1.0, 1.0,
    1.0, -1.0, -1.0,

    // Top
    1.0, 1.0, 1.0,
    -1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0,
    1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0,
    1.0, 1.0, -1.0
};

[shader("vertex")]
AtmosphereVertexOutput vertex_main(uint vertex_id : SV_VertexID) {
    AtmosphereVertexOutput output;
    float3 position = SKYBOX_NONINDEXED_POSITIONS[vertex_id];
    output.view_direction = position;
    
    float4 screen_space_pos = mul(uniforms.clip_from_skybox, float4(position, 1.0));
    output.clip_out = screen_space_pos.xyww;
    return output;
}

[shader("fragment")]
float4 fragment_main(AtmosphereVertexOutput input) : SV_Target0 {
    float3 view_direction = normalize(input.view_direction);

    return float4(sample_fake_sky(
        uniforms,
        view_direction,
        global_textures[uniforms.sunzenith_idx],
        global_textures[uniforms.viewzenith_idx],
        global_textures[uniforms.sunview_idx]
    ), 1.0);
}