
static const float PI = acos(-1);

static const uint MAX_DIRECTIONAL_LIGHTS = 4;
static const uint SHADOW_CASCADE_COUNT = 6;
static const uint CASCADE_DISTANCE_ARRAY_ELEMENTS = (SHADOW_CASCADE_COUNT + 1) * MAX_DIRECTIONAL_LIGHTS;

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

//Variables that are constant for a given frame
struct FrameUniforms {
    float4x4 clip_from_world;
    float4x4 clip_from_view;
    float4x4 view_from_world;
    float4x4 clip_from_skybox;
    float4x4 clip_from_screen;
    float4x4 sun_shadow_matrices[SHADOW_CASCADE_COUNT * MAX_DIRECTIONAL_LIGHTS];
    float4 camera_position;
    // float4 sun_direction;
    // float4 sun_irradiance;
    DirectionalLight directional_lights[MAX_DIRECTIONAL_LIGHTS];
    uint directional_light_count;
    uint sun_shadowmap_idx;
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
    float pad0;
    float pad1;
    float pad2;
    float pad3;
    float4 sun_shadow_distances[CASCADE_DISTANCE_ARRAY_ELEMENTS / 4 + 1];
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
    float3 direction;
    float pad0;
    float3 irradiance;
    float pad1;
};