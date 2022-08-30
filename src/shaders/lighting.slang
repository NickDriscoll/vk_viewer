import structs;

static const float PI = 3.14159265359;

// Everything is from https://learnopengl.com/PBR/Theory

float3 fresnel_schlick(float3 base_reflectivity, float3 halfway, float3 view)
{
    float quintic_term = clamp(1.0 - dot(halfway, view), 0.0, 1.0);
    return base_reflectivity +
           (float3(1.0) - base_reflectivity) * quintic_term * quintic_term * quintic_term * quintic_term * quintic_term;
}

// Trowbridge-Reitz
// Approximation of roughness at a point
float NDFggxtr(float3 normal, float3 halfway, float roughness)
{
    float thedot = max(dot(normal, halfway), 0.0);
    float r2 = roughness * roughness;
    float squared_term = thedot * thedot * (r2 - 1.0) + 1.0;
    return r2 / (PI * squared_term * squared_term);
}

float schlick_ggx(float3 normal, float3 view, float roughness)
{
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0; // Direct
    // float k = roughness * roughness / 2.0; //IBL
    float thedot = max(dot(normal, view), 0.0);
    return thedot / (thedot * (1.0 - k) + k);
}

float geometry_smith(float3 normal, float3 view, float3 light_direction, float roughness)
{
    float ggx1 = schlick_ggx(normal, view, roughness);
    float ggx2 = schlick_ggx(normal, light_direction, roughness);
    return ggx1 * ggx2;
}

float3 f_lambert(float3 color)
{
    return color / PI;
}

struct PBRParams
{
    float3 albedo;
    float3 shading_normal;
    float3 geometry_normal;
    float3 view;
    float3 light_direction;
    float3 light_radiance;
    float roughness;
    float metalness;
    float ambient_occlusion;
}

float3 pbr_mettalic_roughness(PBRParams params)
{
    float ambient_occlusion = params.ambient_occlusion;

    float3 halfway = normalize(params.view + params.light_direction);
    float3 ks = fresnel_schlick(float3(0.04), halfway, params.view);
    float3 kd = float3(1.0) - ks;
    kd *= 1.0 - params.metalness;

    float shading_light_dot = max(dot(params.shading_normal, params.light_direction), 0.0);

    // Cook-Torrence specular term evaluation
    float D = NDFggxtr(params.shading_normal, halfway, params.roughness);
    float3 F = ks;
    float G = geometry_smith(params.shading_normal, params.view, params.light_direction, params.roughness);
    float3 numerator = D * F * G;
    float denominator = 4.0 * max(dot(params.shading_normal, params.view), 0.0) * shading_light_dot + 0.0001;
    float3 cook_torrence = numerator / denominator;

    // Luminance from the sun
    float3 Lo = (kd * f_lambert(params.albedo) + cook_torrence) * params.light_radiance * shading_light_dot;
    float sun_contribution = 1.0 - smoothstep(0.0, -0.05, params.light_direction.z);
    Lo *= sun_contribution;

    return Lo;
}
