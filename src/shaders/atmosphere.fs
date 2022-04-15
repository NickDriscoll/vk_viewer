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
    float time;
};

void main() {
    //frag_color = vec4(sin(time) * 0.5 + 0.5, 0.5, cos(time) * 0.5 + 0.5, 1.0);
    vec3 view_direction = normalize(f_view_direction);    
    vec3 final_color = -1.0 * view_direction * 0.5 + 0.5;

    float sun_likeness = max(0.0, dot(view_direction, sun_direction));
    final_color += smoothstep(mix(1.0, 0.99, 0.5), 1.0, sun_likeness);


    frag_color = vec4(final_color, 1.0);
}