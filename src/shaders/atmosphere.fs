#version 430 core

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
};

const vec3 ZENITH = vec3(0.0, 0.0, 1.0);

void main() {
    vec3 view_direction = normalize(f_view_direction);
    float viewzenith_dot = view_direction.z;


    float color_param = dot(sun_direction, ZENITH);
    vec3 sky_color = (1.0 - color_param) * vec3(1.0, 0.0, 0.0) + color_param * vec3(0.0, 0.0, 1.0);
    vec3 final_color = max(0.0, viewzenith_dot) * sky_color;

    float sun_likeness = max(0.0, dot(view_direction, sun_direction));
    final_color += smoothstep(mix(1.0, 0.99, 0.5), 1.0, sun_likeness);

    frag_color = vec4(final_color, 1.0);
}