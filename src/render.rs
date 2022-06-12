use core::slice::Iter;
use crate::vkutil::VirtualGeometry;
use crate::*;

#[derive(Debug)]
pub struct Material {
    pub color_idx: u32,
    pub normal_idx: u32
}

pub struct DrawCall {
    pub geometry_idx: usize,
    pub pipeline: vk::Pipeline,
    pub instance_count: u32,
    pub first_instance: u32
}

//This is a struct that contains mesh and material data
//In other words, the data required to draw a specific kind of thing
pub struct DrawData {
    pub geometry: VirtualGeometry,
    pub material_idx: u32
}

pub struct InstanceData {
    pub world_from_model: glm::TMat4<f32>,
    pub normal_matrix: glm::TMat4<f32>
}

impl InstanceData {
    pub fn new(world_from_model: glm::TMat4<f32>) -> Self {
        let normal_matrix = glm::mat4_to_mat3(&world_from_model);
        let normal_matrix = glm::transpose(&glm::mat3_to_mat4(&glm::affine_inverse(normal_matrix)));

        InstanceData {
            world_from_model,
            normal_matrix
        }
    }
}

pub struct Renderer {
    models: OptionVec<DrawData>,
    drawlist: Vec<DrawCall>,
    instance_data: Vec<InstanceData>
}

impl Renderer {
    pub fn get_instance_data(&self) -> &Vec<InstanceData> {
        &self.instance_data
    }

    pub fn new() -> Self {
        Renderer {
            models: OptionVec::new(),
            drawlist: Vec::new(),
            instance_data: Vec::new()
        }
    }

    pub fn with_capacity(size: usize) -> Self {
        Renderer {
            models: OptionVec::with_capacity(size),
            drawlist: Vec::with_capacity(size),
            instance_data: Vec::with_capacity(size)
        }
    }

    pub fn register_model(&mut self, data: DrawData) -> usize {
        self.models.insert(data)
    }

    pub fn get_model(&self, idx: usize) -> &Option<DrawData> {
        &self.models[idx]
    }

    pub fn queue_drawcall(&mut self, model_idx: usize, pipeline: vk::Pipeline, transforms: &[glm::TMat4<f32>]) {
        if let None = &self.models[model_idx] {
            tfd::message_box_ok("No model at supplied index", &format!("No model loaded at index {}", model_idx), tfd::MessageBoxIcon::Error);
            return;
        }

        let instance_count = transforms.len() as u32;
        let first_instance = self.instance_data.len() as u32;

        for t in transforms {
            let instance_data = InstanceData::new(*t);
            self.instance_data.push(instance_data);
        }
        let drawcall = DrawCall {
            geometry_idx: model_idx,
            pipeline,
            instance_count,
            first_instance
        };

        self.drawlist.push(drawcall);
    }

    pub fn drawlist_iter(&self) -> Iter<DrawCall> {
        self.drawlist.iter()
    }

    pub fn reset(&mut self) {
        self.drawlist.clear();
        self.instance_data.clear();
    }
}

pub struct FrameUniforms {
    pub clip_from_screen: glm::TMat4<f32>,
    pub clip_from_world: glm::TMat4<f32>,
    pub clip_from_view: glm::TMat4<f32>,
    pub view_from_world: glm::TMat4<f32>,
    pub clip_from_skybox: glm::TMat4<f32>,
    pub camera_position: glm::TVec4<f32>,
    pub sun_direction: glm::TVec4<f32>,
    pub sun_radiance: glm::TVec3<f32>,
    pub time: f32,
    pub stars_threshold: f32, // modifies the number of stars that are visible
	pub stars_exposure: f32   // modifies the overall strength of the stars
}
