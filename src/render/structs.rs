use core::slice::Iter;
use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::Index;
use slotmap::{SlotMap, new_key_type};
use render::{vkdevice, VulkanGraphicsDevice};
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

pub struct VertexInputConfiguration {
    pub binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub attribute_descriptions: Vec<vk::VertexInputAttributeDescription>    
}

impl VertexInputConfiguration {
    pub fn empty() -> Self {
        VertexInputConfiguration {
            binding_descriptions: Vec::new(),
            attribute_descriptions: Vec::new()
        }
    }
}

pub struct PipelineCreateInfo {
    pipeline_layout: vk::PipelineLayout,
    vertex_input: VertexInputConfiguration,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    multisample_state: vk::PipelineMultisampleStateCreateInfo,
    dynamic_state: Vec<vk::DynamicState>,
    viewport_state: vk::PipelineViewportStateCreateInfo,
    depthstencil_state: vk::PipelineDepthStencilStateCreateInfo,
    color_blend_attachment_state: vk::PipelineColorBlendAttachmentState,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    render_pass: vk::RenderPass
}

#[derive(Default)]
pub struct GraphicsPipelineBuilder {
    pub dynamic_state_enables: [vk::DynamicState; 2],
    pub input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo,
    pub rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    pub color_blend_attachment_state: vk::PipelineColorBlendAttachmentState,
    pub viewport_state: vk::PipelineViewportStateCreateInfo,
    pub depthstencil_state: vk::PipelineDepthStencilStateCreateInfo,
    pub multisample_state: vk::PipelineMultisampleStateCreateInfo,
    pub pipeline_layout: vk::PipelineLayout,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    render_pass: vk::RenderPass
}

impl GraphicsPipelineBuilder {
    pub fn new() -> Self {
        let dynamic_state_enables = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_enable: vk::TRUE,
            line_width: 1.0,
            ..Default::default()
        };

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::from_raw(0xF),   //All components
            blend_enable: vk::TRUE,
            alpha_blend_op: vk::BlendOp::ADD,
            color_blend_op: vk::BlendOp::ADD,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            src_alpha_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            p_scissors: ptr::null(),
            p_viewports: ptr::null(),
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        GraphicsPipelineBuilder {
            dynamic_state_enables,
            input_assembly_state,
            rasterization_state,
            color_blend_attachment_state,
            depthstencil_state: depth_stencil_state,
            multisample_state,
            viewport_state,
            ..Default::default()
        }
    }

    pub fn init(render_pass: vk::RenderPass, pipeline_layout: vk::PipelineLayout) -> Self {
        let mut tmp = Self::new();
        tmp.pipeline_layout = pipeline_layout;
        tmp.render_pass = render_pass;
        tmp
    }

    pub fn build_info(self) -> PipelineCreateInfo {
        let vertex_input = VertexInputConfiguration::empty();

        PipelineCreateInfo {
            pipeline_layout: self.pipeline_layout,
            vertex_input,
            input_assembly: self.input_assembly_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            dynamic_state: Vec::from(self.dynamic_state_enables),
            viewport_state: self.viewport_state,
            depthstencil_state: self.depthstencil_state,
            color_blend_attachment_state: self.color_blend_attachment_state,
            shader_stages: self.shader_stages.clone(),
            render_pass: self.render_pass
        }
    }

    pub fn set_cull_mode(self, flags: vk::CullModeFlags) -> Self {
        let mut t = self;
        t.rasterization_state.cull_mode = flags;
        t
    }

    pub fn set_front_face(self, front_face: vk::FrontFace) -> Self {
        let mut t = self;
        t.rasterization_state.front_face = front_face;
        t
    }

    pub fn set_depth_test(self, set_it: vk::Bool32) -> Self {
        let mut t = self;
        t.depthstencil_state.depth_test_enable = set_it;
        t
    }

    pub fn set_shader_stages(self, stages: Vec<vk::PipelineShaderStageCreateInfo>) -> Self {
        GraphicsPipelineBuilder {
            shader_stages: stages,
            ..self
        }
    }

    pub unsafe fn create_pipelines(vk: &mut VulkanGraphicsDevice, infos: &[PipelineCreateInfo]) -> Vec<vk::Pipeline> {
        let mut vertex_input_configs = Vec::with_capacity(infos.len());
        let mut vertex_input_states = Vec::with_capacity(infos.len());
        let mut dynamic_states = Vec::with_capacity(infos.len());
        let mut color_blend_attachment_states = Vec::with_capacity(infos.len());
        let mut real_create_infos = Vec::with_capacity(infos.len());
        for i in 0..infos.len() {
            let info = &infos[i];

            vertex_input_configs.push(&info.vertex_input);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: vertex_input_configs[i].binding_descriptions.len() as u32,
                p_vertex_binding_descriptions: vertex_input_configs[i].binding_descriptions.as_ptr(),
                vertex_attribute_description_count: vertex_input_configs[i].attribute_descriptions.len() as u32,
                p_vertex_attribute_descriptions: vertex_input_configs[i].attribute_descriptions.as_ptr(),
                ..Default::default()
            };
            vertex_input_states.push(vertex_input_state);

            let dynamic_state = vk::PipelineDynamicStateCreateInfo {
                p_dynamic_states: info.dynamic_state.as_ptr(),
                dynamic_state_count: info.dynamic_state.len() as u32,
                ..Default::default()
            };
            dynamic_states.push(dynamic_state);

            let color_blend = vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 1,
                p_attachments: &info.color_blend_attachment_state,
                logic_op_enable: vk::FALSE,
                logic_op: vk::LogicOp::NO_OP,
                blend_constants: [0.0; 4],
                ..Default::default()
            };
            color_blend_attachment_states.push(color_blend);

            let create_info = vk::GraphicsPipelineCreateInfo {
                layout: info.pipeline_layout,
                p_vertex_input_state: &vertex_input_states[i],
                p_input_assembly_state: &info.input_assembly,
                p_rasterization_state: &info.rasterization_state,
                p_color_blend_state: &color_blend_attachment_states[i],
                p_multisample_state: &info.multisample_state,
                p_dynamic_state: &dynamic_states[i],
                p_viewport_state: &info.viewport_state,
                p_depth_stencil_state: &info.depthstencil_state,
                p_stages: info.shader_stages.as_ptr(),
                stage_count: info.shader_stages.len() as u32,
                render_pass: info.render_pass,
                ..Default::default()
            };
            real_create_infos.push(create_info);
        }

        vk.device.create_graphics_pipelines(vk::PipelineCache::null(), &real_create_infos, vkdevice::MEMORY_ALLOCATOR).unwrap()
    }

}

#[derive(Debug)]
pub struct FreeList<T> {
    list: OptionVec<T>,
    size: u64,
    updated: bool
}

impl<T> FreeList<T> {
    pub fn new() -> Self {
        FreeList {
            list: OptionVec::new(),
            size: 0,
            updated: false
        }
    }

    pub fn with_capacity(size: usize) -> Self {
        FreeList {
            list: OptionVec::with_capacity(size),
            size: size as u64,
            updated: false
        }
    }

    pub fn len(&self) -> usize { self.list.len() }

    pub fn size(&self) -> u64 { self.size }

    pub fn insert(&mut self, item: T) -> usize {
        self.updated = true;
        self.list.insert(item)
    }

    pub fn remove(&mut self, idx: usize) -> Option<T> {
        self.updated = true;
        self.list.delete(idx)
    }

	pub fn get_element(&mut self, index: usize) -> Option<&T> {
		self.list.get_element(index)
	}

	pub fn get_mut_element(&mut self, index: usize) -> Option<&mut T> {
		self.list.get_mut_element(index)
	}

    pub fn was_updated(&mut self) -> bool {
        let r = self.updated;
        self.updated = false;
        r
    }
}

impl<T> Index<usize> for FreeList<T> {
    type Output = Option<T>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.list[idx]
    }
}
