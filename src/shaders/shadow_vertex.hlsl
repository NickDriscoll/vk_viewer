static const uint SHADOW_CASCADE_COUNT = 6;
static const uint CASCADE_DISTANCE_ARRAY_ELEMENTS = (SHADOW_CASCADE_COUNT + 1) / 5 + 1;

//Standard input to a vertex shader
struct ModelVertex {
    uint instance_id : SV_InstanceID;
    uint id : SV_VertexID;
};

struct ShadowVertexOutput {
    float4 clip_out : SV_Position;
    float2 uv : UV;
};

struct InstanceData {
    float4x4 world_from_model;
    float4x4 normal_matrix;
};

// Variables that are constant for a given frame
struct FrameUniforms {
    float4x4 clip_from_world;
    float4x4 clip_from_view;
    float4x4 view_from_world;
    float4x4 clip_from_skybox;
    float4x4 clip_from_screen;
    float4x4 sun_shadow_matrices[SHADOW_CASCADE_COUNT];
    float4 camera_position;
    float4 sun_direction;
    float4 sun_radiance;
    uint sun_shadowmap_idx;
    float time;
    float stars_threshold;
    float stars_exposure;
    float fog_density;
    uint sunzenith_idx;
    uint viewzenith_idx;
    uint sunview_idx;
    float exposure;
    float sun_intensity;
    float ambient_factor;
    float _pad0;
    float4 sun_shadow_distances[CASCADE_DISTANCE_ARRAY_ELEMENTS];
};

[[vk::binding(0, 0)]]
ConstantBuffer<FrameUniforms> uniforms;
[[vk::binding(1, 0)]]
StructuredBuffer<InstanceData> instance_buffer;
[[vk::binding(3, 0)]]
StructuredBuffer<float4> vertex_positions;
[[vk::binding(6, 0)]]
StructuredBuffer<float2> vertex_uvs;

struct PushConstants
{
    uint _material_idx;
    uint position_buffer_offset;
    uint uv_buffer_offset;
};

[[vk::push_constant]]
PushConstants push_constants;

ShadowVertexOutput main(ModelVertex vertex, int view_id : SV_ViewID) {
    float4x4 model_matrix = instance_buffer[vertex.instance_id].world_from_model;
    float3 position = vertex_positions[vertex.id + push_constants.position_buffer_offset].xyz;
    float3 world_position = mul(model_matrix, float4(position, 1.0)).xyz;

    ShadowVertexOutput o;
    o.clip_out = mul(uniforms.sun_shadow_matrices[view_id], float4(world_position, 1.0));
    o.uv = vertex_uvs[vertex.id + push_constants.uv_buffer_offset];

    return o;
}