struct InstanceData {
    float4x4 world_from_model;
    float4x4 normal_matrix;
}

//Standard input to a vertex shader
struct VertexInput {
    float3 position : POSITION;
    float3 tangent : TANGENT;
    float3 bitangent: BITANGENT;
    float3 normal: NORMAL;
    float2 uvs: UVS;
    uint instance_id : SV_InstanceID;
}

//Standard output from a vertex shader
struct VertexOutput {
    float4 clip_out : SV_Position;
    float3 f_position;
    float3 f_normal;
    float2 f_uvs;
    float3 f_tangent;
    float3 f_bitangent;
}

//Variables that are constant for a given frame
struct FrameUniforms {
    float4x4 clip_from_screen;
    float4x4 clip_from_world;
    float4x4 clip_from_view;
    float4x4 view_from_world;
    float4x4 clip_from_skybox;
    float4 sun_position;
    float4 sun_direction;
    float4 sun_radiance;
    float time;
    float stars_threshold;
    float stars_exposure;
}
