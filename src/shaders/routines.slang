import structs;

// Apply fog like GL_EXP2
float3 exp2_fog(FrameUniforms uniforms, float3 frag_position, float3 in_color, float3 fog_color) {
    //Compute fog factor
    float real_depth = length(uniforms.camera_position.xyz - frag_position);
    float base = 0.001 * uniforms.fog_density * real_depth;
    float fog_factor = exp(-(base * base));

    return lerp(fog_color, in_color, fog_factor);
}