#version 430 core
#extension GL_EXT_nonuniform_qualifier: enable

layout (location = 0) in vec3 f_view_direction;

layout (location = 0) out vec4 frag_color;

#include "../frame_uniforms.sl"

layout(set = 0, binding = 1) uniform sampler2D global_textures2D[];
layout(set = 0, binding = 1) uniform samplerCube global_cubemaps[];

layout(push_constant) uniform TexIndices {
    uint sunzenith_idx;
    uint viewzenith_idx;
    uint sunview_idx;
};

// 3D Gradient noise from: https://www.shadertoy.com/view/Xsl3Dl
vec3 hash( vec3 p ) {// replace this by something better
	p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
			  dot(p,vec3(269.5,183.3,246.1)),
			  dot(p,vec3(113.5,271.9,124.6)));

	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec3 p ) {
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);

    return mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// from Unity's black body Shader Graph node
vec3 Unity_Blackbody_float(float Temperature) {
    vec3 color = vec3(255.0, 255.0, 255.0);
    color.x = 56100000. * pow(Temperature,(-3.0 / 2.0)) + 148.0;
    color.y = 100.04 * log(Temperature) - 623.6;
    if (Temperature > 6500.0) color.y = 35200000.0 * pow(Temperature,(-3.0 / 2.0)) + 184.0;
    color.z = 194.18 * log(Temperature) - 1448.6;
    color = clamp(color, 0.0, 255.0)/255.0;
    if (Temperature < 1000.0) color *= Temperature/1000.0;
    return color;
}

void main() {
    vec3 view_direction = normalize(f_view_direction);
    float sunzenith_dot = sun_direction.z * 0.5 + 0.5;
    float sunview_dot = dot(sun_direction, view_direction) * 0.5 + 0.5;
    float viewzenith_dot = view_direction.z * 0.5 + 0.5;

    vec3 base_color = texture(global_textures2D[sunzenith_idx], vec2(sunzenith_dot, 0.5)).rgb;

    vec3 viewzenith_color = texture(global_textures2D[viewzenith_idx], vec2(sunzenith_dot, 0.5)).rgb;
    viewzenith_color *= pow(1.0 - viewzenith_dot, 4.0);

    vec3 sunview_color = texture(global_textures2D[sunview_idx], vec2(sunzenith_dot, 0.5)).rgb;
    sunview_color *= pow(sunview_dot, 1.0);

    // Stars computation:
	float stars = pow(clamp(noise(view_direction * 200.0f), 0.0f, 1.0f), stars_threshold) * stars_exposure;
    float star_fact = mix(0.4, 1.4, noise(view_direction * 100.0f + vec3(time)));
	stars *= clamp(star_fact, 0.1, 1.4); // time based flickering
 
    // star color by randomized temperature
    float stars_temperature = noise(view_direction * 150.0) * 0.5 + 0.5;
    vec3 stars_color = stars * Unity_Blackbody_float(mix(1500.0, 65000.0, pow(stars_temperature,4.0)));
    stars_color *= 1.0 - smoothstep(-0.1, 0.0, sun_direction.z);
    
    vec3 final_color = base_color + viewzenith_color + sunview_color + stars_color;

    final_color += smoothstep(mix(1.0, 0.99, 0.25), 1.0, sunview_dot);

    frag_color = vec4(final_color, 1.0);
    //frag_color = vec4(vec3(star_fact), 1.0);
}