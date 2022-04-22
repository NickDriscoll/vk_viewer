#version 430 core
#extension GL_EXT_nonuniform_qualifier: enable

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

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

layout(set = 0, binding = 1) uniform sampler2D global_textures[];

struct Material {
    uint color_map_index;
    
};

layout (set = 0, binding = 2) readonly buffer MaterialData {
    Material global_materials[];
};

layout(push_constant) uniform Indices {
    uint color_map_index;
};

void main() {
    vec3 normal = normalize(f_normal);
    float sun_contribution = max(0.05, dot(normal, sun_direction));

    vec3 base_color = texture(global_textures[color_map_index], f_uv).rgb;
    vec3 norm_color = normal * 0.5 + 0.5;

    for (int i = 0; i < 4; i++) {
        if (f_position.z < 10.0 * (i + 1)) {
            
            break;
        }
    }

    frag_color = vec4(sun_contribution * base_color, 1.0);
}
