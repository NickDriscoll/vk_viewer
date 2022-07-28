import structs;

// Bindings
ConstantBuffer<FrameUniforms> uniforms;

[[vk_binding(8, 0)]]
Sampler2D global_textures[];

[[vk::push_constant]]
cbuffer PushConstants {
    uint sunzenith_idx;
    uint viewzenith_idx;
    uint sunview_idx;
}

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

// 3D Gradient noise from: https://www.shadertoy.com/view/Xsl3Dl
float3 hash( float3 p ) {// replace this by something better
	p = float3( dot(p,float3(127.1,311.7, 74.7)),
			  dot(p,float3(269.5,183.3,246.1)),
			  dot(p,float3(113.5,271.9,124.6)));

	return -1.0 + 2.0*frac(sin(p)*43758.5453123);
}

float noise( in float3 p ) {
    float3 i = floor(p);
    float3 f = frac(p);

    float3 u = f*f*(3.0-2.0*f);

    return lerp( lerp( lerp( dot( hash( i + float3(0.0,0.0,0.0) ), f - float3(0.0,0.0,0.0) ), 
                          dot( hash( i + float3(1.0,0.0,0.0) ), f - float3(1.0,0.0,0.0) ), u.x),
                     lerp( dot( hash( i + float3(0.0,1.0,0.0) ), f - float3(0.0,1.0,0.0) ), 
                          dot( hash( i + float3(1.0,1.0,0.0) ), f - float3(1.0,1.0,0.0) ), u.x), u.y),
                lerp( lerp( dot( hash( i + float3(0.0,0.0,1.0) ), f - float3(0.0,0.0,1.0) ), 
                          dot( hash( i + float3(1.0,0.0,1.0) ), f - float3(1.0,0.0,1.0) ), u.x),
                     lerp( dot( hash( i + float3(0.0,1.0,1.0) ), f - float3(0.0,1.0,1.0) ), 
                          dot( hash( i + float3(1.0,1.0,1.0) ), f - float3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// from Unity's black body Shader Graph node
float3 Unity_Blackbody_float(float Temperature) {
    float3 color = float3(255.0, 255.0, 255.0);
    color.x = 56100000. * pow(Temperature,(-3.0 / 2.0)) + 148.0;
    color.y = 100.04 * log(Temperature) - 623.6;
    if (Temperature > 6500.0) color.y = 35200000.0 * pow(Temperature,(-3.0 / 2.0)) + 184.0;
    color.z = 194.18 * log(Temperature) - 1448.6;
    color = clamp(color, 0.0, 255.0)/255.0;
    if (Temperature < 1000.0) color *= Temperature/1000.0;
    return color;
}

[shader("fragment")]
float4 fragment_main(AtmosphereVertexOutput input) : SV_Target0 {
    float3 view_direction = normalize(input.view_direction);
    float sunzenith_dot = uniforms.sun_direction.z * 0.5 + 0.5;
    float sunview_dot = dot(uniforms.sun_direction.xyz, view_direction) * 0.5 + 0.5;
    float viewzenith_dot = view_direction.z * 0.5 + 0.5;

    float3 base_color = global_textures[sunzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;

    float3 viewzenith_color = global_textures[viewzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
    viewzenith_color *= pow(1.0 - viewzenith_dot, 4.0);

    float3 sunview_color = global_textures[sunview_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
    sunview_color *= pow(sunview_dot, 4.0);

    //Used to bring out the stars/dim the sun
    float nighttime_factor = smoothstep(-0.1, 0.1, uniforms.sun_direction.z);
    
    // Stars computation:
	float stars = pow(clamp(noise(view_direction * 200.0f), 0.0f, 1.0f), uniforms.stars_threshold) * uniforms.stars_exposure;
    float star_fact = lerp(0.4, 1.4, noise(view_direction * 100.0f + float3(uniforms.time)));
	stars *= clamp(star_fact, 0.1, 1.4); // time based flickering
 
    // star color by randomized temperature
    float stars_temperature = noise(view_direction * 150.0) * 0.5 + 0.5;
    float3 stars_color = stars * Unity_Blackbody_float(lerp(1500.0, 65000.0, pow(stars_temperature,4.0)));
    stars_color *= 1.0 - nighttime_factor;
    
    float3 final_color = base_color + viewzenith_color + sunview_color + stars_color;

    final_color += nighttime_factor * smoothstep(lerp(1.0, 0.99, 0.25), 1.0, sunview_dot);

    return float4(final_color, 1.0);
}