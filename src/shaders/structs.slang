struct InstanceData {
    float4x4 world_from_model;
    float4x4 normal_matrix;
}

//Standard input to a vertex shader
struct ModelVertex {
    float3 position : POSITION;
    float3 tangent : TANGENT;
    float3 bitangent: BITANGENT;
    float3 normal: NORMAL;
    float2 uv: UVS;
    uint instance_id : SV_InstanceID;
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

struct AtmosphereVertexOutput {
    float4 clip_out: SV_Position;
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
}

struct Material {
    uint color_map_index;
    uint normal_map_index;
};
