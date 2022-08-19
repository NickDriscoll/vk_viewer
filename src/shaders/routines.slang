import structs;

/*
//2D simplex noise
#version 120
vec3 permute(vec3 x) {
  return mod(((x ∗ 34.0) + 1.0) ∗ x, 289.0);
}

vec3 taylorInvSqrt(vec3 r) {
  return 1.79284291400159 − 0.85373472095314 ∗ r;
}

float snoise(vec2 P) {
    const vec2 C = vec2(0.211324865405187134, //(3.0−sqrt(3.0))/6.0;
                        0.366025403784438597);//0.5∗(sqrt(3.0)−1.0);
    //Firstcorner
    vec2 i = floor(P + dot(P, C.yy));
    vec2 x0 = P − i + dot(i, C.xx);

    //Other corners
    vec2 i1;
    i1.x = step(x0.y, x0.x);//1.0ifx0.x>x0.y,else0.0
    i1.y = 1.0 − i1.x;
    //x1=x0−i1+1.0∗C.xx;x2=x0−1.0+2.0∗C.xx;
    vec4 x12 = x0.xyxy + vec4(C.xx, C.xx ∗ 2.0 − 1.0);
    x12.xy −= i1;

    //Permutations
    i = mod(i, 289.0);//Avoidtruncationinpolynomialevaluation
    vec3 p = permute(permute(i.y+vec3(0.0,i1.y,1.0))
              + i.x + vec3(0.0,i1.x,1.0));
    
    //Circularly symmetric blending kernel
    vec3 m = max(0.5 − vec3(dot(x0, x0), dot(x12.xy, x12.xy),
                 dot(x12.zw, x12.zw)),0.0);
    m = m ∗ m;
    m = m ∗ m;

    //Gradients from 41 points on a line, mapped on to a diamond
    vec3 x = fract(p ∗ (1.0 / 41.0)) ∗ 2.0 − 1.0;
    vec3 gy = abs(x) − 0.5;
    vec3 ox = floor(x + 0.5); //round(x) is a GLSL1.30 feature
    vec3 gx = x − ox;
    //Normalise gradients implicitly by scaling m
    m ∗= taylorInvSqrt(gx ∗ gx + gy ∗ gy);
    //Compute final noise value at P
    vec3 g;
    g.x = gx.x ∗ x0.x + gy.x ∗ x0.y;
    g.yz = gx.yz ∗ x12.xz + gy.yz ∗ x12.yw;
    //Scale output to span range[−1,1]
    //(scalingfactordeterminedbyexperiments)
    return130.0∗dot(m,g);
}
*/


// Apply fog like GL_EXP2
float3 exp2_fog(FrameUniforms uniforms, float3 frag_position, float3 in_color, float3 fog_color) {
    //Compute fog factor
    float real_depth = length(uniforms.camera_position.xyz - frag_position);
    float base = 0.001 * uniforms.fog_density * real_depth;
    float fog_factor = exp(-(base * base));

    return lerp(fog_color, in_color, fog_factor);
}

// 3D Gradient noise from: https://www.shadertoy.com/view/Xsl3Dl
float3 hash(float3 p) { // replace this by something better
    p = float3(dot(p, float3(127.1, 311.7, 74.7)),
               dot(p, float3(269.5, 183.3, 246.1)),
               dot(p, float3(113.5, 271.9, 124.6)));

    return -1.0 + 2.0 * frac(sin(p) * 43758.5453123);
}

float noise(in float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);

    float3 u = f * f * (3.0 - 2.0 * f);

    return lerp(lerp(lerp(dot(hash(i + float3(0.0, 0.0, 0.0)), f - float3(0.0, 0.0, 0.0)),
                          dot(hash(i + float3(1.0, 0.0, 0.0)), f - float3(1.0, 0.0, 0.0)), u.x),
                     lerp(dot(hash(i + float3(0.0, 1.0, 0.0)), f - float3(0.0, 1.0, 0.0)),
                          dot(hash(i + float3(1.0, 1.0, 0.0)), f - float3(1.0, 1.0, 0.0)), u.x), u.y),
                lerp(lerp(dot(hash(i + float3(0.0, 0.0, 1.0)), f - float3(0.0, 0.0, 1.0)),
                          dot(hash(i + float3(1.0, 0.0, 1.0)), f - float3(1.0, 0.0, 1.0)), u.x),
                     lerp(dot(hash(i + float3(0.0, 1.0, 1.0)), f - float3(0.0, 1.0, 1.0)),
                          dot(hash(i + float3(1.0, 1.0, 1.0)), f - float3(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

// from Unity's black body Shader Graph node
float3 Unity_Blackbody_float(float Temperature) {
    float3 color = float3(255.0, 255.0, 255.0);
    color.x = 56100000. * pow(Temperature, (-3.0 / 2.0)) + 148.0;
    color.y = 100.04 * log(Temperature) - 623.6;
    if (Temperature > 6500.0) color.y = 35200000.0 * pow(Temperature, (-3.0 / 2.0)) + 184.0;
    color.z = 194.18 * log(Temperature) - 1448.6;
    color = clamp(color, 0.0, 255.0) / 255.0;
    if (Temperature < 1000.0) color *= Temperature / 1000.0;
    return color;
}

float3 sample_fake_sky(FrameUniforms uniforms, float3 view_direction, Sampler2D sun_zenith, Sampler2D view_zenith, Sampler2D sun_view) {
    float sunzenith_dot = uniforms.sun_direction.z * 0.5 + 0.5;
    float sunview_dot = dot(uniforms.sun_direction.xyz, view_direction) * 0.5 + 0.5;
    float viewzenith_dot = view_direction.z * 0.5 + 0.5;

    float3 base_color = sun_zenith.Sample(float2(sunzenith_dot, 0.5)).rgb;

    float3 viewzenith_color = view_zenith.Sample(float2(sunzenith_dot, 0.5)).rgb;
    viewzenith_color *= pow(1.0 - viewzenith_dot, 3.0);

    float3 sunview_color = sun_view.Sample(float2(sunzenith_dot, 0.5)).rgb;
    sunview_color *= pow(sunview_dot, 6.0);

    // Used to bring out the stars/dim the sun
    float nighttime_factor = smoothstep(-0.1, 0.1, uniforms.sun_direction.z);

    // Stars computation
    float stars = pow(clamp(noise(view_direction * 200.0f), 0.0f, 1.0f), uniforms.stars_threshold) * uniforms.stars_exposure;
    float star_fact = lerp(0.4, 1.4, noise(view_direction * 100.0f + float3(uniforms.time)));
    stars *= clamp(star_fact, 0.1, 1.4); // time based flickering

    // star color by randomized temperature
    float stars_temperature = noise(view_direction * 150.0) * 0.5 + 0.5;
    float3 stars_color = stars * Unity_Blackbody_float(lerp(1500.0, 65000.0, pow(stars_temperature, 4.0)));
    stars_color *= 1.0 - nighttime_factor;

    float3 final_color = base_color + viewzenith_color + sunview_color + stars_color;

    final_color += nighttime_factor * smoothstep(lerp(1.0, 0.99, 0.25), 1.0, sunview_dot);

    return final_color;
}