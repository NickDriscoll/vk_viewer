import structs;

/*
//2D simplex noise from paper "Efficient computational noise in GLSL" 12/10/2011

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

float3 permute(float3 x) {
    return ((x * 34.0) + 1.0) * x % 289.0;
}

float3 taylorInvSqrt(float3 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}

// 2D simplex noise from paper "Efficient computational noise in GLSL" 12/10/2011
float simplex_noise(float2 P) {
    static const float2 C = float2(0.211324865405187134,  //(3.0−sqrt(3.0))/6.0;
                                 0.366025403784438597); // 0.5∗(sqrt(3.0)−1.0);

    // First corner
    float2 i = floor(P + dot(P, C.yy));
    float2 x0 = P - i + dot(i, C.xx); 

    // Other corners
    float2 i1;
    i1.x = step(x0.y, x0.x); // 1.0 if x0.x > x0.y, else 0.0
    i1.y = 1.0 - i1.x;
    // x1 = x0 − i1 + 1.0 ∗ C.xx; x2 = x0 − 1.0 + 2.0 ∗ C.xx;
    float4 x12 = x0.xyxy + float4(C.xx, C.xx * 2.0 - 1.0);
    x12.xy -= i1;

    // Permutations
    i %= 289.0;
    float3 p = permute(
        permute(i.y + float3(0.0, i1.y, 1.0)) +
        i.x +
        float3(0.0, i1.x, 1.0)
    );

    // Circularly symmetric blending kernel
    float3 m = max(0.5 - float3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;

    // Gradients from 41 points on a line, mapped onto a diamond
    float3 x = frac(p * (1.0 / 41.0)) * 2.0 - 1.0;
    float3 gy = abs(x) - 0.5;
    float3 ox = floor(x + 0.5); // round(x) is a GLSL1.30 feature
    float3 gx = x - ox;

    // Normalise gradients implicitly by scaling m
    m *= taylorInvSqrt(gx * gx + gy * gy);

    // Compute final noise value at P
    float3 g;
    g.x = gx.x * x0.x + gy.x * x0.y;
    g.yz = gx.yz * x12.xz + gy.yz * x12.yw;

    // Scale output to span range[−1,1]
    // (scaling factor determined experimentally)
    return 130.0 * dot(m, g);
}

//Modifies roughness based on normal variance
float toksvigs_method(float3 sampled_normal, float roughness) {
    float averaged_normal_length = length(sampled_normal);
    //float ggx_roughness = 2.0 / (roughness * roughness) - 2.0; //Equivalence for GGX
    float ggx_roughness = roughness;

    return averaged_normal_length * ggx_roughness / (averaged_normal_length + ggx_roughness * (1.0 - averaged_normal_length));
}

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

float3 sample_sky_stars(FrameUniforms uniforms, float3 view_direction) {
    float nighttime_factor = smoothstep(-0.1, 0.1, uniforms.sun_direction.z);

    // Stars computation
    float stars = pow(clamp(noise(view_direction * 200.0f), 0.0f, 1.0f), uniforms.stars_threshold) * uniforms.stars_exposure;
    float star_fact = lerp(0.4, 1.4, noise(view_direction * 100.0f + float3(uniforms.time)));
    stars *= clamp(star_fact, 0.1, 1.4); // time based flickering

    // star color by randomized temperature
    float stars_temperature = noise(view_direction * 150.0) * 0.5 + 0.5;
    float3 stars_color = stars * Unity_Blackbody_float(lerp(1500.0, 65000.0, pow(stars_temperature, 4.0)));
    stars_color *= 1.0 - nighttime_factor;

    return stars_color;
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

    float nighttime_factor = smoothstep(-0.1, 0.1, uniforms.sun_direction.z);
    float3 final_color = base_color + viewzenith_color + sunview_color;

    final_color += nighttime_factor * smoothstep(lerp(1.0, 0.99, 0.25), 1.0, sunview_dot);

    return final_color;
}

float determine_shadowed(float3 f_shadow_pos, float3 normal, float3 light_direction, Sampler2DArray shadow_map, int cascade) {
    float bias;
    // bias = 0.00125;
    // bias = 0.0;
    bias = 0.00125 * (1.0 - max(0.0, dot(normal, light_direction))) + 0.00125;

    float3 sample_uv = float3(f_shadow_pos.xy, (float)cascade);
    
    float sampled_depth = shadow_map.Sample(sample_uv).r;
    return sampled_depth + bias < f_shadow_pos.z ? 1.0 : 0.0;
}

float cascaded_shadow_factor(float3 adj_shadow_space_pos, Sampler2DArray shadow_map, int shadow_cascade, float3 normal, float3 light_direction) {
    // Compute how shadowed if we are potentially shadowed
    float shadow = 0.0;
    if (shadow_cascade > -1) {
        if (true) {
            //Do PCF
            //Average the NxN block of shadow texels centered at this pixel
            int bound = 2;
            float2 texel_size;
            float el;
            float l;
            shadow_map.GetDimensions(0, texel_size.x, texel_size.y, el, l);
            texel_size = 1.0 / texel_size;
            for (int x = -bound; x <= bound; x++) {
                for (int y = -bound; y <= bound; y++) {
                    shadow += determine_shadowed(float3(adj_shadow_space_pos.xy + float2(x, y) * texel_size, adj_shadow_space_pos.z), normal, light_direction, shadow_map, shadow_cascade);
                }
            }
            float s = 2.0 * bound + 1.0;
            shadow /= s * s; //Total number of texels sampled: (2*bound + 1)^2
        } else {
            shadow = determine_shadowed(adj_shadow_space_pos.xyz, normal, light_direction, shadow_map, shadow_cascade);
        }
    }

    //float shadow = shadow_map.SampleCmp(float3(adj_shadow_space_pos.xy, (float)shadow_cascade), adj_shadow_space_pos.z);
    
    return 1.0 - shadow;
}

// Returns shadow factor in x and cascade index in y
float2 compute_cascade_shadowing(
    Sampler2DArray shadow_map,
    float3 normal,
    float3 light_direction,
    float4 shadow_space_positions[SHADOW_CASCADE_COUNT],
    float4 cascade_distances[CASCADE_DISTANCE_ARRAY_ELEMENTS / 4 + 1],
    float clip_space_z
) {
    //Determine which cascade the fragment is in
    float3 adj_shadow_space_pos;
    int shadow_cascade = -1;
    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        float cascade_distance = cascade_distances[(i + 1) / 4][(i + 1) % 4];
        if (clip_space_z < cascade_distance) {
            adj_shadow_space_pos = float3(float2(shadow_space_positions[i].xy * 0.5 + 0.5), shadow_space_positions[i].z);
            if (!(
                adj_shadow_space_pos.z < 0.0 ||
                adj_shadow_space_pos.z > 1.0 ||
                adj_shadow_space_pos.x < 0.0 ||
                adj_shadow_space_pos.x > 1.0 ||
                adj_shadow_space_pos.y < 0.0 ||
                adj_shadow_space_pos.y > 1.0
            )) {
                shadow_cascade = i;
            }
            break;
        }
    }

    // Compute sun shadow factor
    float shadow_factor = cascaded_shadow_factor(adj_shadow_space_pos, shadow_map, shadow_cascade, normal, light_direction);
    return float2(shadow_factor, float(shadow_cascade));
}

struct ray_t {
    float3 origin;
    float3 direction;
};

struct sphere_t {
    float3 origin;
    float radius;
    int material;
};

struct plane_t {
    float3 direction;
    float distance;
    int material;
};

float3x3 rotate_around_x(float angle_degrees)
{
    float angle = radians(angle_degrees);
    float _sin = sin(angle);
    float _cos = cos(angle);
    return float3x3(1, 0, 0, 0, _cos, -_sin, 0, _sin, _cos);
}

ray_t get_primary_ray(
    float3 cam_local_point,
    inout float3 cam_origin,
    inout float3 cam_look_at
) {
    float3 fwd = normalize(cam_look_at - cam_origin);
    float3 up = float3(0, 1, 0);
    float3 right = cross(up, fwd);
    up = cross(fwd, right);

    ray_t r = {
        cam_origin,
        normalize(fwd + up * cam_local_point.y + right * cam_local_point.x)
    };

    // ray_t r = _begin(ray_t)
    //           cam_origin,
    //       normalize(fwd + up * cam_local_point.y + right * cam_local_point.x)
    //           _end;

    return r;
}

// scattering coefficients at sea level (m)
static const float3 betaR = float3(5.5e-6, 13.0e-6, 22.4e-6); // Rayleigh
static const float3 betaM = float3(21e-6);                    // Mie

// scale height (m)
// thickness of the atmosphere if its density were uniform
static const float hR = 7994.0; // Rayleigh
static const float hM = 1200.0; // Mie

float rayleigh_phase_func(float mu)
{
    return
            3.0 * (1.0 + mu * mu)
    / //------------------------
           (16.0 * PI);
}

// Henyey-Greenstein phase function factor [-1, 1]
// represents the average cosine of the scattered directions
// 0 is isotropic scattering
// > 1 is forward scattering, < 1 is backwards
static const float g = 0.76;
float henyey_greenstein_phase_func(float mu)
{
    return
                        (1.0 - g * g)
    / //---------------------------------------------
           ((4.0 * PI) * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
}

// Schlick Phase Function factor
// Pharr and  Humphreys [2004] equivalence to g above
static const float k = 1.55 * g - 0.55 * (g * g * g);
float schlick_phase_func(float mu)
{
    return
                    (1.0 - k * k)
    / //-------------------------------------------
           (4.0 * PI * (1.0 + k * mu) * (1.0 + k * mu));
}

static const float earth_radius = 6360e3;             // (m)
static const float atmosphere_radius = 6420e3; // (m)

static const sphere_t atmosphere = {
    float3(0.0, 0.0, 0.0),
    atmosphere_radius,
    0
};

bool isect_sphere(ray_t ray, sphere_t sphere, out float t0, out float t1)
{
    float3 rc = sphere.origin - ray.origin;
    float radius2 = sphere.radius * sphere.radius;
    float tca = dot(rc, ray.direction);
    float d2 = dot(rc, rc) - tca * tca;
    if (d2 > radius2) return false;
    float thc = sqrt(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;

    return true;
}

bool get_sun_light(
    ray_t ray,
    in out float optical_depthR,
    in out float optical_depthM
) {
    const int num_samples = 8;
    float t0, t1;
    isect_sphere(ray, atmosphere, t0, t1);

    float march_pos = 0.0;
    float march_step = t1 / float(num_samples);

    for (int i = 0; i < num_samples; i++) {
        float3 s = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
        float height = length(s) - earth_radius;
        if (height < 0.0)
            return false;

        optical_depthR += exp(-height / hR) * march_step;
        optical_depthM += exp(-height / hM) * march_step;

        march_pos += march_step;
    }

    return true;
}

float3 get_incident_light(ray_t ray, float3 sun_dir, float sun_power) {
    const int num_samples = 16;

    // "pierce" the atmosphere with the viewing ray
    float t0, t1;
    bool did_isect = isect_sphere(ray, atmosphere, t0, t1);
    if (!did_isect) {
        return float3(0.6);
    }

    float march_step = t1 / float(num_samples);

    // cosine of angle between view and light directions
    float mu = dot(ray.direction, sun_dir);

    // Rayleigh and Mie phase functions
    // A black box indicating how light is interacting with the material
    // Similar to BRDF except
    // * it usually considers a single angle
    //   (the phase angle between 2 directions)
    // * integrates to 1 over the entire sphere of directions
    float phaseR = rayleigh_phase_func(mu);
    float phaseM = henyey_greenstein_phase_func(mu);

    // optical depth (or "average density")
    // represents the accumulated extinction coefficients
    // along the path, multiplied by the length of that path
    float optical_depthR = 0.0;
    float optical_depthM = 0.0;

    float3 sumR = float3(0.0);
    float3 sumM = float3(0.0);
    float march_pos = 0.0;

    for (int i = 0; i < num_samples; i++) {
        float3 s = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
        float height = length(s) - earth_radius;

        // integrate the height scale
        float hr = exp(-height / hR) * march_step;
        float hm = exp(-height / hM) * march_step;
        optical_depthR += hr;
        optical_depthM += hm;

        // gather the sunlight
        ray_t light_ray = {
            s,
            sun_dir
        };
        float optical_depth_lightR = 0.0;
        float optical_depth_lightM = 0.0;
        bool overground = get_sun_light(light_ray, optical_depth_lightR, optical_depth_lightM);

        if (overground) {
            float3 tau =
                betaR * (optical_depthR + optical_depth_lightR) +
                betaM * 1.1 * (optical_depthM + optical_depth_lightM);
            float3 attenuation = exp(-tau);

            sumR += hr * attenuation;
            sumM += hm * attenuation;
        }

        march_pos += march_step;
    }

    return 20.0 * (sumR * phaseR * betaR + sumM * phaseM * betaM);
}