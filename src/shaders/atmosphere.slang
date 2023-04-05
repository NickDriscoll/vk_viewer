import structs;
import routines;
import bindings.frame_uniforms;
import bindings.global_textures;

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

[[vk::push_constant]]
cbuffer AtmospherePushConstants
{
    uint sun_idx;
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

static const float earth_radius = 6360e3;      // (m)
static const float atmosphere_radius = 6420e3; // (m)

static const sphere_t atmosphere = {
    float3(0.0, 0.0, 0.0),
    atmosphere_radius,
    0
};

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

float3 gather_atmosphere_irradiance(ray_t ray, float3 sun_dir, float3 sun_irradiance) {
    const int num_samples = 16;

    // "pierce" the atmosphere with the viewing ray
    float t0, t1;
    bool did_isect = isect_sphere(ray, atmosphere, t0, t1);
    if (!did_isect) {
        return float3(0.6);
    }

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
    float march_step = t1 / float(num_samples);

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

    return sun_irradiance * (sumR * phaseR * betaR + sumM * phaseM * betaM);
}

[shader("vertex")]
AtmosphereVertexOutput vertex_main(uint vertex_id : SV_VertexID) {
    AtmosphereVertexOutput output;
    float3 position = SKYBOX_NONINDEXED_POSITIONS[vertex_id];
    output.view_direction = position;

    output.clip_out = mul(uniforms.clip_from_skybox, float4(position, 1.0));
    output.clip_out.z = 0.0;
    return output;
}

[shader("fragment")]
float4 fragment_main(AtmosphereVertexOutput input) : SV_Target0 {
    float3 view_direction = normalize(input.view_direction);
    DirectionalLight sun = uniforms.directional_lights[sun_idx];

    float3 sky_sample;
    if (uniforms.real_sky) {
        ray_t ray = {
            float3(0.0, 0.0, earth_radius) + uniforms.camera_position.xyz,
            view_direction
        };
        sky_sample = gather_atmosphere_irradiance(ray, sun.direction.xyz, sun.irradiance.rgb);
    } else {
        sky_sample = sun.irradiance.rgb * sample_fake_sky(
            sun.direction.xyz,
            view_direction,
            global_textures[uniforms.sunzenith_idx],
            global_textures[uniforms.viewzenith_idx],
            global_textures[uniforms.sunview_idx]
        );
    }

    float3 stars_color = sample_sky_stars(uniforms, view_direction, sun.direction.z);
    float sun_closeness = dot(view_direction, sun.direction.xyz);
    float3 sun_color = normalize(sky_sample) * sun.irradiance * smoothstep(0.99950, 1.0, sun_closeness);
    sun_color = max(sun_color, float3(0.0));
    float3 final_color = sun_color + sky_sample + stars_color;

    return float4(final_color, 1.0);
}
