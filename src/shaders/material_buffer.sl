struct Material {
    uint color_map_index;
    uint normal_map_index;
};
layout (set = 0, binding = 3) readonly buffer MaterialData {
    Material global_materials[];
};