use core::slice::Iter;
use std::cmp::Ordering;
use std::hash::Hash;
use slotmap::{SlotMap, new_key_type};
use render::vkdevice;

use crate::*;
pub trait UniqueID {
    fn id(&self) -> u64;
}

pub struct DeferredDelete {
    pub key: PrimitiveKey,
    pub frames_til_deletion: u32
}

//SlotMap key types used by Renderer
new_key_type! { pub struct ModelKey; }
new_key_type! { pub struct PrimitiveKey; }
new_key_type! { pub struct ImageKey; }
new_key_type! { pub struct PositionBufferBlockKey; }

//1:1 with shader struct
#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct MaterialData {
    pub base_color: [f32; 4],
    pub base_roughness: f32,
    pub base_metalness: f32,
    pub color_idx: u32,
    pub normal_idx: u32,
    pub metal_roughness_idx: u32,
    pub emissive_idx: u32,
    pad0: u32,
    pad1: u32,
}

#[derive(Debug)]
pub struct Material {
    pub pipeline: vk::Pipeline,
    pub base_color: [f32; 4],
    pub base_roughness: f32,
    pub base_metalness: f32,
    pub color_idx: Option<u32>,
    pub normal_idx: Option<u32>,
    pub metal_roughness_idx: Option<u32>,
    pub emissive_idx: Option<u32>
}

impl Material {
    pub fn data(&self, renderer: &Renderer) -> MaterialData {
        let color_idx = match self.color_idx {
            Some(idx) => { idx }
            None => { renderer.default_color_idx }
        };
        let normal_idx = match self.normal_idx {
            Some(idx) => { idx }
            None => { renderer.default_normal_idx }
        };
        let metal_roughness_idx = match self.metal_roughness_idx {
            Some(idx) => { idx }
            None => { renderer.default_metal_roughness_idx }
        };
        let emissive_idx = match self.emissive_idx {
            Some(idx) => { idx }
            None => { renderer.default_emissive_idx }
        };

        MaterialData {
            base_color: self.base_color,
            base_roughness: self.base_roughness,
            base_metalness: self.base_metalness,
            color_idx,
            normal_idx,
            metal_roughness_idx,
            emissive_idx,
            pad0: 0,
            pad1: 0
        }
    }
}

pub struct DesiredDraw {
    pub model_key: ModelKey,
    pub world_transforms: Vec<glm::TMat4<f32>>
}

pub struct DrawCall {
    pub primitive_key: PrimitiveKey,
    pub pipeline: vk::Pipeline,
    pub instance_count: u32,
    pub first_instance: u32
}

impl PartialEq for DrawCall {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
    }
}
impl Eq for DrawCall {}

impl PartialOrd for DrawCall {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DrawCall {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.pipeline < other.pipeline {
            Ordering::Less
        } else if self.pipeline == other.pipeline {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }
}

pub enum ShadowType {
    OpaqueCaster,
    NonCaster
}

//This is a struct that contains mesh and material data
//In other words, the data required to drive a single draw call
pub struct Primitive {
    pub shadow_type: ShadowType,
    pub index_buffer: vkdevice::GPUBuffer,
    pub index_count: u32,
    pub position_offset: u32,
    pub tangent_offset: u32,
    pub normal_offset: u32,
    pub uv_offset: u32,
    pub material_idx: u32
}

#[derive(Clone)]
pub struct Model {
    pub id: u64,            //Just the hash of the asset's name
    pub primitive_keys: Vec<PrimitiveKey>
}

impl UniqueID for Model {
    fn id(&self) -> u64 {
        self.id
    }
}

#[repr(C)]
pub struct InstanceData {
    pub world_transform: glm::TMat4<f32>,
    pub normal_matrix: glm::TMat4<f32>
}

impl InstanceData {
    pub fn new(world_from_model: glm::TMat4<f32>) -> Self {
        //TL;DR: use cofactor instead of transpose(inverse(world_from_model)) to compute normal matrix
        //https://github.com/graphitemaster/normals_revisited
        fn cofactor(src: &[f32], dst: &mut [f32]) {
            fn minor(m: &[f32], r0: usize, r1: usize, r2: usize, c0: usize, c1: usize, c2: usize) -> f32 {
                return m[4*r0+c0] * (m[4*r1+c1] * m[4*r2+c2] - m[4*r2+c1] * m[4*r1+c2]) -
                       m[4*r0+c1] * (m[4*r1+c0] * m[4*r2+c2] - m[4*r2+c0] * m[4*r1+c2]) +
                       m[4*r0+c2] * (m[4*r1+c0] * m[4*r2+c1] - m[4*r2+c0] * m[4*r1+c1]);
            }
        
            dst[ 0] =  minor(src, 1, 2, 3, 1, 2, 3);
            dst[ 1] = -minor(src, 1, 2, 3, 0, 2, 3);
            dst[ 2] =  minor(src, 1, 2, 3, 0, 1, 3);
            dst[ 3] = -minor(src, 1, 2, 3, 0, 1, 2);
            dst[ 4] = -minor(src, 0, 2, 3, 1, 2, 3);
            dst[ 5] =  minor(src, 0, 2, 3, 0, 2, 3);
            dst[ 6] = -minor(src, 0, 2, 3, 0, 1, 3);
            dst[ 7] =  minor(src, 0, 2, 3, 0, 1, 2);
            dst[ 8] =  minor(src, 0, 1, 3, 1, 2, 3);
            dst[ 9] = -minor(src, 0, 1, 3, 0, 2, 3);
            dst[10] =  minor(src, 0, 1, 3, 0, 1, 3);
            dst[11] = -minor(src, 0, 1, 3, 0, 1, 2);
            dst[12] = -minor(src, 0, 1, 2, 1, 2, 3);
            dst[13] =  minor(src, 0, 1, 2, 0, 2, 3);
            dst[14] = -minor(src, 0, 1, 2, 0, 1, 3);
            dst[15] =  minor(src, 0, 1, 2, 0, 1, 2);
        }
        
        let mut normal_matrix: glm::TMat4<f32> = glm::identity();
        cofactor(world_from_model.as_slice(), normal_matrix.as_mut_slice());

        InstanceData {
            world_transform: world_from_model,
            normal_matrix
        }
    }
}

//Values that will be uniform over a particular simulation frame
#[derive(Default)]
#[repr(C)]
pub struct EnvironmentUniforms {
    pub clip_from_world: glm::TMat4<f32>,
    pub clip_from_view: glm::TMat4<f32>,
    pub view_from_world: glm::TMat4<f32>,
    pub clip_from_skybox: glm::TMat4<f32>,
    pub clip_from_screen: glm::TMat4<f32>,
    pub sun_shadow_matrices: [glm::TMat4<f32>; CascadedShadowMap::CASCADE_COUNT],
    pub camera_position: glm::TVec4<f32>,
    pub sun_direction: glm::TVec4<f32>,
    pub sun_irradiance: glm::TVec4<f32>,
    pub sun_shadowmap_idx: u32,
    pub time: f32,
    pub stars_threshold: f32, // modifies the number of stars that are visible
	pub stars_exposure: f32,  // modifies the overall strength of the stars
    pub fog_density: f32,
    pub sunzenith_idx: u32,
    pub viewzenith_idx: u32,
    pub sunview_idx: u32,
    pub exposure: f32,
    pub ambient_factor: f32,
    pub real_sky: f32,
    pub pad0: f32,
    pub sun_shadow_distances: [f32; CascadedShadowMap::CASCADE_COUNT + 1],
}

pub struct CascadedShadowMap {
    framebuffer: vk::Framebuffer,
    image: vk::Image,
    image_view: vk::ImageView,
    format: vk::Format,
    texture_index: u32,
    resolution: u32,
    clip_distances: [f32; Self::CASCADE_COUNT + 1],
    view_distances: [f32; Self::CASCADE_COUNT + 1]
}

impl CascadedShadowMap {
    pub const CASCADE_COUNT: usize = 6;

    pub fn clip_distances(&self) -> [f32; Self::CASCADE_COUNT + 1] { self.clip_distances }
    pub fn view_distances(&self) -> [f32; Self::CASCADE_COUNT + 1] { self.view_distances }

    pub fn update_view_distances(&mut self, clipping_from_view: &glm::TMat4<f32>, view_distances: [f32; Self::CASCADE_COUNT + 1]) {
        for i in 0..view_distances.len() {
            let p = clipping_from_view * glm::vec4(0.0, 0.0, view_distances[i], 1.0);
            self.clip_distances[i] = p.z;
        }
        self.view_distances = view_distances;
    }

    pub fn framebuffer(&self) -> vk::Framebuffer { self.framebuffer }
    pub fn image(&self) -> vk::Image { self.image }
    pub fn resolution(&self) -> u32 { self.resolution }
    pub fn texture_index(&self) -> u32 { self.texture_index }
    pub fn view(&self) -> vk::ImageView { self.image_view }

    pub fn new(
        vk: &mut vkdevice::VulkanGraphicsDevice,
        renderer: &mut Renderer,
        resolution: u32,
        clipping_from_view: &glm::TMat4<f32>,
        render_pass: vk::RenderPass
    ) -> Self {
        let format = vk::Format::D32_SFLOAT;

        let allocation;
        let image = unsafe {
            let extent = vk::Extent3D {
                width: resolution,
                height: resolution,
                depth: 1
            };

            let create_info = vk::ImageCreateInfo {
                queue_family_index_count: 1,
                p_queue_family_indices: [vk.queue_family_index].as_ptr(),
                flags: vk::ImageCreateFlags::empty(),
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent,
                mip_levels: 1,
                array_layers: Self::CASCADE_COUNT as u32,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };

            let depth_image = vk.device.create_image(&create_info, vkdevice::MEMORY_ALLOCATOR).unwrap();
            allocation = vkdevice::allocate_image_memory(vk, depth_image);
            depth_image
        };

        let image_view = unsafe {
            let view_info = vk::ImageViewCreateInfo {
                image,
                format,
                view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                components: vkdevice::COMPONENT_MAPPING_DEFAULT,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: Self::CASCADE_COUNT as u32
                },
                ..Default::default()
            };
            vk.device.create_image_view(&view_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
        };

        //Create framebuffer
        let framebuffer = unsafe {
            let attachments = [image_view];
            let fb_info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: resolution,
                height: resolution,
                layers: 1,
                ..Default::default()
            };
            
            vk.device.create_framebuffer(&fb_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
        };

        //Manually picking the cascade distances because math is hard
        //The shadow cascade distances are negative bc they are in view space
        let mut view_distances = [0.0; Self::CASCADE_COUNT + 1];
        view_distances[0] = -(Renderer::NEAR_CLIP_DISTANCE);
        view_distances[1] = -(Renderer::NEAR_CLIP_DISTANCE + 10.0);
        view_distances[2] = -(Renderer::NEAR_CLIP_DISTANCE + 30.0);
        view_distances[3] = -(Renderer::NEAR_CLIP_DISTANCE + 73.0);
        view_distances[4] = -(Renderer::NEAR_CLIP_DISTANCE + 123.0);
        view_distances[5] = -(Renderer::NEAR_CLIP_DISTANCE + 222.0);
        view_distances[6] = -(Renderer::NEAR_CLIP_DISTANCE + 500.0);

        //Compute the clip space distances
        let mut clip_distances = [0.0; Self::CASCADE_COUNT + 1];
        for i in 0..view_distances.len() {
            let p = clipping_from_view * glm::vec4(0.0, 0.0, view_distances[i], 1.0);
            clip_distances[i] = p.z;
        }

        let gpu_image = vkdevice::GPUImage {
            image,
            view: Some(image_view),
            width: resolution,
            height: resolution,
            mip_count: 1,
            format,
            layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            sampler: renderer.shadow_sampler,
            allocation
        };
        
        let texture_index = renderer.global_images.insert(gpu_image) as u32;

        CascadedShadowMap {
            framebuffer,
            image,
            image_view,
            format,
            texture_index,
            resolution,
            clip_distances,
            view_distances
        }
    }

    pub fn compute_shadow_cascade_matrices(
        &self,
        light_direction: &glm::TVec3<f32>,
        v_mat: &glm::TMat4<f32>,
        projection: &glm::TMat4<f32>
    ) -> [glm::TMat4<f32>; Self::CASCADE_COUNT] {
        let mut out_mats = [glm::identity(); Self::CASCADE_COUNT];

        let shadow_view = glm::look_at(&(light_direction * 20.0), &glm::zero(), &glm::vec3(0.0, 0.0, 1.0));    
        let shadow_from_view = shadow_view * glm::affine_inverse(*v_mat);
        let fovx = f32::atan(1.0 / projection[0]);
        let fovy = f32::atan(1.0 / projection[5]);
    
        //Loop computes the shadow matrices for this frame
        for i in 0..Self::CASCADE_COUNT {
            //Near and far distances for this sub-frustum
            let z0 = self.view_distances[i];
            let z1 = self.view_distances[i + 1];
    
            //Computing the view-space coords of the sub-frustum vertices
            let x0 = -z0 * f32::tan(fovx);
            let x1 = z0 * f32::tan(fovx);
            let x2 = -z1 * f32::tan(fovx);
            let x3 = z1 * f32::tan(fovx);
            let y0 = -z0 * f32::tan(fovy);
            let y1 = z0 * f32::tan(fovy);
            let y2 = -z1 * f32::tan(fovy);
            let y3 = z1 * f32::tan(fovy);
    
            //The extreme vertices of the sub-frustum
            let shadow_view_space_points = [
                shadow_from_view * glm::vec4(x0, y0, z0, 1.0),
                shadow_from_view * glm::vec4(x1, y0, z0, 1.0),
                shadow_from_view * glm::vec4(x0, y1, z0, 1.0),
                shadow_from_view * glm::vec4(x1, y1, z0, 1.0),                                        
                shadow_from_view * glm::vec4(x2, y2, z1, 1.0),
                shadow_from_view * glm::vec4(x3, y2, z1, 1.0),
                shadow_from_view * glm::vec4(x2, y3, z1, 1.0),
                shadow_from_view * glm::vec4(x3, y3, z1, 1.0)                                        
            ];
    
            //Determine the view-space boundaries of the orthographic projection
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for point in shadow_view_space_points.iter() {
                if max_x < point.x { max_x = point.x; }
                if min_x > point.x { min_x = point.x; }
                if max_y < point.y { max_y = point.y; }
                if min_y > point.y { min_y = point.y; }
            }
    
            let projection_depth = 500.0;
            let shadow_projection = glm::ortho_rh_zo(
                min_x, max_x, min_y, max_y, -projection_depth, projection_depth
            );
            let shadow_projection = glm::mat4(
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ) * shadow_projection;
    
            out_mats[i] = shadow_projection * shadow_view;
        }
        out_mats
    }
}

pub struct SunLight {
    pub pitch: f32,
    pub yaw: f32,
    pub pitch_speed: f32,
    pub yaw_speed: f32,
    pub irradiance: glm::TVec3<f32>,
    pub shadow_map: CascadedShadowMap
}
