import structs;

// Everything is from https://learnopengl.com/PBR/Theory
// https://www.pbr-book.org/3ed-2018/Color_and_Radiometry/Radiometry

float3 fresnel_schlick(float3 base_reflectivity, float3 halfway, float3 view)
{
    float quintic_term = max(1.0 - dot(halfway, view), 0.0);
    return base_reflectivity + (float3(1.0) - base_reflectivity) * quintic_term * quintic_term * quintic_term * quintic_term * quintic_term;
}

// Trowbridge-Reitz
// Statistically approximates the relative surface area of microfacets exactly aligned to the halfway vector
float NDFggxtr(float3 normal, float3 halfway, float roughness)
{
    float thedot = max(dot(normal, halfway), 0.0);
    float r2 = roughness * roughness;
    float squared_term = thedot * thedot * (r2 - 1.0) + 1.0;
    float denom = max(PI * squared_term * squared_term, 0.00001);
    return r2 / denom;
}

float schlick_ggx(float3 normal, float3 view, float roughness)
{
    float r = roughness + 1.0;

    float k = r * r / 8.0; // Direct
    // float k = roughness * roughness / 2.0; //IBL

    float thedot = max(dot(normal, view), 0.000001);
    return thedot / (thedot * (1.0 - k) + k);
}

float geometry_smith(float3 normal, float3 view, float3 light_direction, float roughness)
{
    return schlick_ggx(normal, view, roughness) * schlick_ggx(normal, light_direction, roughness);
}

//Lambertian BRDF
float3 f_lambert(float3 color)
{
    return color / PI;
}

struct PBRSurface
{
    float3 albedo;
    float3 shading_normal;
    float3 geometry_normal;
    float3 view;
    float roughness;
    float metalness;
}

struct DirectionalLight {
    float3 direction;
    float3 irradiance;
}

//Returns the outgoing radiance caused by the directional light hitting the surface
float3 directional_lighting(PBRSurface surface, DirectionalLight light)
{
    float3 halfway = normalize(surface.view + light.direction);
    float3 F0 = lerp(float3(0.04), surface.albedo, surface.metalness);
    float3 ks = fresnel_schlick(F0, halfway, surface.view);
    float3 kd = float3(1.0) - ks;
    kd *= 1.0 - surface.metalness;

    float shading_light_dot = max(dot(surface.shading_normal, light.direction), 0.0);
    float roughness = surface.roughness;

    // Cook-Torrence specular term evaluation
    float D = NDFggxtr(surface.shading_normal, halfway, roughness);
    float3 F = ks;
    float G = geometry_smith(surface.shading_normal, surface.view, light.direction, roughness);
    float3 numerator = D * F * G;
    float denominator = 4.0 * max(dot(surface.shading_normal, surface.view), 0.0) * shading_light_dot + 0.0001;
    float3 cook_torrence = numerator / denominator;

    // Compute outgoing radiance
    float3 Lo = (kd * f_lambert(surface.albedo) + cook_torrence) * light.irradiance * shading_light_dot;

    // Directional lights can't contribute when they're under the world
    Lo *= 1.0 - smoothstep(0.0, -0.05, light.direction.z);

    return Lo;
}
