import structs;

    // Apply fog like GL_EXP2
float3 exp2_fog(FrameUniforms uniforms, Sampler2D global_textures[], float3 frag_position, float3 in_color, float3 fog_color) {
    //Compute fog factor
    float sunzenith_dot = uniforms.sun_direction.z * 0.5 + 0.5;
    float real_depth = length(uniforms.camera_position.xyz - frag_position);
    float base = 0.001 * uniforms.fog_density * real_depth;
    float fog_factor = exp(-(base * base));

    return lerp(fog_color, in_color, fog_factor);
}