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

pub struct DrawSystem {
    models: OptionVec<DrawData>,
    drawlist: Vec<DrawCall>,
    transforms: Vec<glm::TMat4<f32>>
}

impl DrawSystem {
    pub fn get_transforms(&self) -> &Vec<glm::TMat4<f32>> {
        &self.transforms
    }

    pub fn new() -> Self {
        DrawSystem {
            models: OptionVec::new(),
            drawlist: Vec::new(),
            transforms: Vec::new()
        }
    }

    pub fn with_capacity(size: usize) -> Self {
        DrawSystem {
            models: OptionVec::with_capacity(size),
            drawlist: Vec::with_capacity(size),
            transforms: Vec::with_capacity(size)
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
        let first_instance = self.transforms.len() as u32;

        for t in transforms {
            self.transforms.push(*t);
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
        self.transforms.clear();
    }
}

pub struct FrameUniforms {
    pub clip_from_screen: glm::TMat4<f32>,
    pub clip_from_world: glm::TMat4<f32>,
    pub clip_from_view: glm::TMat4<f32>,
    pub view_from_world: glm::TMat4<f32>,
    pub sun_direction: glm::TVec3<f32>,
    pub time: f32,
    pub stars_threshold: f32, // modifies the number of stars that are visible
	pub stars_exposure: f32   // modifies the overall strength of the stars
}
