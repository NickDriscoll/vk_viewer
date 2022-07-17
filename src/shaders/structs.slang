struct InstanceData {
    float4x4 world_from_model;
    float4x4 normal_matrix;
}

//Standard input to a vertex shader
struct ModelVertex {
    float3 position;
    float3 tangent;
    float3 bitangent;
    float3 normal;
    float2 uv;
    uint instance_id : SV_InstanceID;
    uint id : SV_VertexID;
}

//Standard output from a vertex shader
struct MeshVertexOutput {
    float4 clip_out : SV_Position;
    float3 position;
    float3 normal;
    float2 uv;
    float3 tangent;
    float3 bitangent;
}

struct AtmosphereVertex {
    float3 position: POSITION;
}

struct AtmosphereVertexOutput
{
    float4 clip_out : SV_Position;
    float3 view_direction;
}

//Variables that are constant for a given frame
struct FrameUniforms {
    float4x4 clip_from_screen;
    float4x4 clip_from_world;
    float4x4 clip_from_view;
    float4x4 view_from_world;
    float4x4 clip_from_skybox;
    float4 camera_position;
    float4 sun_direction;
    float4 sun_radiance;
    float time;
    float stars_threshold;
    float stars_exposure;
    float prog_inter;
}

struct Material {
    float4 base_color;
    uint color_map_index;
    uint normal_map_index;
    uint pad0;
    uint pad1;
};
