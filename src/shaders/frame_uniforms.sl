layout (std140, set = 0, binding = 0) readonly uniform FrameData {
    mat4 clip_from_screen;
    mat4 clip_from_world;
    mat4 clip_from_view;
    mat4 view_from_world;
    mat4 clip_from_skybox;
    vec3 sun_direction;
    float time;
	float stars_threshold; // modifies the number of stars that are visible
	float stars_exposure; // modifies the overall strength of the stars
};

const float LIGHTING_MIN = 0.15;