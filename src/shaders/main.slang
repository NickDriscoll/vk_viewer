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

struct InstanceData {
    float4x4 world_from_model;
    float4x4 normal_matrix;
}

struct VertexOutput {
    float4 clip_out : SV_Position;
    float3 f_position;
    float3 f_normal;
    float2 f_uvs;
    float3 f_tangent;
    float3 f_bitangent;
}

ConstantBuffer<FrameUniforms> uniforms;
Texture2D global_textures[];
StructuredBuffer<InstanceData> instance_buffer;

[shader("vertex")]
VertexOutput main(
    float3 position : POSITION,
    float3 tangent : TANGENT,
    float3 bitangent: BITANGENT,
    float3 normal: NORMAL,
    float2 uvs: UVS,
    uint instance_id : SV_InstanceID
) {
    float4x4 model_matrix = instance_buffer[instance_id].world_from_model;
    float4x4 normal_matrix = instance_buffer[instance_id].normal_matrix;

    float3 world_position = mul(model_matrix, float4(position, 1.0)).xyz;
    
    float3 world_tangent = normalize(mul(normal_matrix, float4(tangent, 1.0)).xyz);
    float3 world_bitangent = normalize(mul(normal_matrix, float4(bitangent, 1.0)).xyz);
    float3 world_normal = normalize(mul(normal_matrix, float4(normal, 1.0)).xyz);

    //Assemble and return output
    VertexOutput output;
    output.clip_out = mul(uniforms.clip_from_world, float4(world_position, 1.0));
    output.f_position = world_position;
    output.f_normal = world_normal;
    output.f_uvs = uvs;
    output.f_tangent = world_tangent;
    output.f_bitangent = world_bitangent;

    return output;
}