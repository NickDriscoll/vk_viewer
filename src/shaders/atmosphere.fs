#version 430 core
#extension GL_EXT_nonuniform_qualifier: enable

layout (location = 0) in vec3 f_view_direction;

layout (location = 0) out vec4 frag_color;

layout (std140, set = 0, binding = 0) readonly uniform FrameData {
    mat4 clip_from_screen;
    mat4 clip_from_world;
    mat4 clip_from_view;
    mat4 view_from_world;
    mat4 clip_from_skybox;
    vec3 sun_direction;
    float pad0;
    vec3 sun_color;
    float time;
};

layout(set = 0, binding = 1) uniform sampler2D global_textures[];

layout(push_constant) uniform TexIndices {
    uint sunzenith_idx;
    uint sunview_idx;
    //uint sunzenith_idx;
    //uint sunzenith_idx;
};

void main() {
    vec3 view_direction = normalize(f_view_direction);
    float sunzenith_dot = max(0.0, sun_direction.z);
    float sunview_dot = max(0.0, dot(sun_direction, view_direction));
    float viewzenith_dot = max(0.0, view_direction.z);

    vec3 base_color = texture(global_textures[sunzenith_idx], vec2(sunzenith_dot, 0.5)).rgb;
    vec3 sunview_color = texture(global_textures[sunview_idx], vec2(sunview_dot, 0.5)).rgb;
    
    vec3 final_color = base_color + sunview_color;

    float sun_likeness = max(0.0, dot(view_direction, sun_direction));
    final_color += smoothstep(mix(1.0, 0.99, 0.5), 1.0, sun_likeness);

    frag_color = vec4(final_color, 1.0);
}