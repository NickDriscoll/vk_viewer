
static const float PI = acos(-1);

static const uint MAX_DIRECTIONAL_LIGHTS = 4;
static const uint SHADOW_CASCADE_COUNT = 6;
static const uint CASCADE_DISTANCE_ARRAY_ELEMENTS = (SHADOW_CASCADE_COUNT + 1) + (4 - ((SHADOW_CASCADE_COUNT + 1) % 4));

struct InstanceData {
    float4x4 world_from_model;
    float4x4 normal_matrix;
}

//Standard input to a vertex shader
struct ModelVertex {
    uint instance_id : SV_InstanceID;
    uint id : SV_VertexID;
}

//Standard output from a vertex shader
struct MeshVertexOutput {
    float4 clip_out : SV_Position;
    float clip_space_z;
    float3 position;
    float3 normal;
    float2 uv;
    float3 tangent;
    float3 bitangent;
    float4 sunshadow_space_positions[SHADOW_CASCADE_COUNT];
}

struct ShadowVertexOutput {
    float4 clip_out : SV_Position;
    float2 uv;
}

struct AtmosphereVertexOutput
{
    float4 clip_out : SV_Position;
    float3 view_direction;
}

struct ImguiVertex {
    float2 position : POSITION;
    float2 uv : UVS;
    float4 color : COLOR;
}

struct ImguiVertexOutput {
    float4 clip_out : SV_Position;
    float2 uv;
    float4 color;
}

struct AtmosphereUniforms {
    float3 sun_direction;
    float stars_exposure;
    float3 sun_radiance;
    float stars_threshold;
}

// Variables that are constant for a given frame
struct FrameUniforms {
    float4x4 clip_from_screen;
    float4x4 clip_from_world;
    float4x4 clip_from_view;
    float4x4 view_from_world;
    float4x4 clip_from_skybox;
    float4 camera_position;
    DirectionalLight directional_lights[MAX_DIRECTIONAL_LIGHTS];
    uint directional_light_count;
    float time;
    float stars_threshold;
    float stars_exposure;
    float fog_density;
    uint sunzenith_idx;
    uint viewzenith_idx;
    uint sunview_idx;
    float exposure;
    float ambient_factor;
    bool real_sky;
    float bloom_strength;
    float3 sky_sample;
    float emissive_exaggeration;
}

struct Camera {
    float4x4 clip_from_world;
    float4x4 clip_from_view;
    float4x4 clip_from_skybox;
    float4x4 view_from_world;
    float3 position;
    float exposure;
}

struct Material {
    float4 base_color;
    float base_roughness;
    float base_metalness;
    uint color_map_index;
    uint normal_map_index;
    uint metal_roughness_index;
    uint emissive_index;
    uint pad0;
    uint pad1;
    float3 emissive_power; // In Watts/m^2/steradian
    uint pad2;
}

struct PBRSurface {
    float3 albedo;
    float3 shading_normal;
    float3 geometry_normal;
    float3 view;
    float roughness;
    float metalness;
}

struct DirectionalLight {
    float4x4 shadow_matrices[SHADOW_CASCADE_COUNT];
    float4 shadow_distances[CASCADE_DISTANCE_ARRAY_ELEMENTS / 4];
    float3 direction;
    uint shadow_map_index;
    float3 irradiance;
    float pad0;
};