import structs;
import lighting;
import routines;
import atmosphere;
import bindings.frame_uniforms;
import bindings.material_buffer;
import bindings.global_textures;

[[vk::push_constant]]
cbuffer PushConstants
{
    uint material_idx;
}

// The MIT License
// Copyright © 2015 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// How to do texture map a 3D object when it doesn't have
// uv coordinates but can't afford full 3D solid texturing.

// The idea is to perform three planar texture projections
// and blend the results based on the alignment of the
// normal vector to each one of the projection directions.

// The technique was invented by Mitch Prater in the early
// 90s, and has been called "Box mapping" or "Rounded cube
// mapping" traditionally, although more recently it has
// become popular in the realtime rendering community and
// rebranded as "triplanar" mapping.

// For a "biplanar" mapping example, visit:
//
// https://www.shadertoy.com/view/ws3Bzf

// "p" point apply texture to
// "n" normal at "p"
// "k" controls the sharpness of the blending in the
//     transitions areas.
// "s" texture sampler
float4 triplanar(in Sampler2D s, in float3 p, in float3 n, in float k) {
    // project+fetch
    static const float sample_scale = 0.05;
    float4 x = s.Sample(p.yz * sample_scale);
    float4 y = s.Sample(p.zx * sample_scale);
    float4 z = s.Sample(p.xy * sample_scale);

    // and blend
    float3 m = pow(abs(n), float3(k));
    return (x * m.x + y * m.y + z * m.z) / (m.x + m.y + m.z);
}

struct MaterialInfo {
    float3 albedo;
    float3 normal;
    float roughness;
    float metalness;
    float ao;
}

MaterialInfo info_from_material(Material mat, float2 uvs, float3 world_pos, float3 geo_normal) {
    MaterialInfo out;

    //float4 sample = global_textures[mat.color_map_index].Sample(uvs);
    float4 sample = triplanar(global_textures[mat.color_map_index], world_pos, geo_normal, 8.0);
    out.albedo = sample.rgb;
    float3 n_sample = 2.0 * triplanar(global_textures[mat.normal_map_index], world_pos, geo_normal, 8.0).xyz - 1.0;
    //float3 n_sample = 2.0 * global_textures[mat.normal_map_index].Sample(uvs).xyz - 1.0;
    out.normal = normalize(n_sample);

    float4 metal_roughness_sample = triplanar(global_textures[mat.metal_roughness_index], world_pos, geo_normal, 8.0);
    //float4 metal_roughness_sample = global_textures[mat.metal_roughness_index].Sample(uvs);

    // out.roughness = toksvigs_method(n_sample, mat.base_roughness * metal_roughness_sample.g);
    out.roughness = mat.base_roughness * metal_roughness_sample.g;
    out.metalness = mat.base_metalness * metal_roughness_sample.b;
    out.ao = metal_roughness_sample.r;

    return out;
}

[shader("fragment")]
float4 fragment_main(MeshVertexOutput input) : SV_Target0
{
    float3 tangent = normalize(input.tangent);
    float3 bitangent = normalize(input.bitangent);
    float3 normal = normalize(input.normal);
    float3 view = normalize(uniforms.camera_position.xyz - input.position);
    float3x3 TBN = float3x3(tangent, bitangent, normal);
    float2 uvs = input.uv;

    Material grass_material = material_buffer[material_idx];
    MaterialInfo grass_info = info_from_material(grass_material, input.uv, input.position, normal);

    Material rock_material = material_buffer[material_idx + 1];
    MaterialInfo rock_info = info_from_material(rock_material, input.uv, input.position, normal);

    float mix_factor = max(0.0, dot(normal, float3(0.0, 0.0, 1.0)));
    mix_factor = smoothstep(0.60, 0.70, mix_factor);

    float3 world_normal = mul(lerp(rock_info.normal, grass_info.normal, mix_factor), TBN);
    float roughness = lerp(rock_info.roughness, grass_info.roughness, mix_factor);
    float metalness = lerp(rock_info.metalness, grass_info.metalness, mix_factor);
    float ao = lerp(rock_info.ao, grass_info.ao, mix_factor);

    PBRSurface params;
    params.albedo = lerp(rock_info.albedo, grass_info.albedo, mix_factor);
    params.shading_normal = world_normal;
    params.geometry_normal = normal;
    params.view = view;
    params.roughness = roughness;
    params.metalness = metalness;

    // Outgoing radiance accumulator
    float3 final_radiance = float3(0.0);

    for (uint i = 0; i < uniforms.directional_light_count; i++) {
        uint light_idx = i;
        DirectionalLight light = uniforms.directional_lights[i];
        float nighttime_factor = smoothstep(-0.1, 0.1, light.direction.z);

        // ray_t ray = {
        //     float3(0.0, 0.0, earth_radius) + input.position,
        //     light.direction
        // };
        // float3 sky_sample = normalize(gather_atmosphere_irradiance(ray, light.direction.xyz, light.irradiance.rgb));
        float3 sky_sample = normalize(uniforms.sky_sample);

        // Compute sun color
        float sunzenith_dot = light.direction.z * 0.5 + 0.5;
        float3 halo_color = float3(1.0);
        float3 sky_color = float3(1.0);
        float3 sun_color;
        if (!uniforms.real_sky) {
            sky_color = global_textures[uniforms.sunzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
            halo_color = global_textures[uniforms.viewzenith_idx].Sample(float2(sunzenith_dot, 0.5)).rgb;
        }
        sun_color = lerp(sky_color, halo_color, 0.2);

        float2 sun_shadowing = compute_cascade_shadowing(
            global_array_textures[light.shadow_map_index],
            normal,
            light.direction.xyz,
            input.sunshadow_space_positions,
            uniforms.directional_lights[i].shadow_distances,
            input.clip_space_z,
            i
        );

        final_radiance += nighttime_factor * sun_shadowing.x * sky_sample * directional_lighting(params, light);

        float3 ambient = float3(uniforms.ambient_factor) * sky_sample * params.albedo * ao;
        final_radiance += ambient;
    }
    
    return float4(final_radiance, 1.0);
}
