use core::slice::Iter;
use std::{convert::TryInto, ffi::c_void};
use ash::vk::DescriptorImageInfo;

use crate::*;

//1:1 with shader struct
#[derive(Clone, Debug)]
#[repr(C)]
pub struct MaterialData {
    pub base_color: [f32; 4],
    pub base_roughness: f32,
    pub color_idx: u32,
    pub normal_idx: u32,
    pub metal_roughness_idx: u32,
    pub emissive_idx: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[derive(Debug)]
pub struct Material {
    pub pipeline: vk::Pipeline,
    pub base_color: [f32; 4],
    pub base_roughness: f32,
    pub color_idx: u32,
    pub normal_idx: u32,
    pub metal_roughness_idx: u32,
    pub emissive_idx: u32
}

impl Material {
    pub fn data(&self) -> MaterialData {
        MaterialData {
            base_color: self.base_color,
            base_roughness: self.base_roughness,
            color_idx: self.color_idx,
            normal_idx: self.normal_idx,
            metal_roughness_idx: self.metal_roughness_idx,
            emissive_idx: self.emissive_idx,
            pad0: 0,
            pad1: 0,
            pad2: 0
        }
    }
}

pub struct DrawCall {
    pub geometry_idx: usize,
    pub pipeline: vk::Pipeline,
    pub instance_count: u32,
    pub first_instance: u32
}

pub enum ShadowType {
    OpaqueCaster,
    NonCaster
}

//This is a struct that contains mesh and material data
//In other words, the data required to draw a specific kind of thing
pub struct Primitive {
    pub shadow_type: ShadowType,
    pub index_buffer: GPUBuffer,
    pub index_count: u32,
    pub position_offset: u32,
    pub tangent_offset: u32,
    pub normal_offset: u32,
    pub uv_offset: u32,
    pub material_idx: u32
}

#[repr(C)]
pub struct InstanceData {
    pub world_from_model: glm::TMat4<f32>,
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
            world_from_model,
            normal_matrix
        }
    }
}

//Values that will be uniform to a particular view. Probably shouldn't be called "FrameUniforms"
#[derive(Default)]
#[repr(C)]
pub struct FrameUniforms {
    pub clip_from_world: glm::TMat4<f32>,
    pub clip_from_view: glm::TMat4<f32>,
    pub view_from_world: glm::TMat4<f32>,
    pub clip_from_skybox: glm::TMat4<f32>,
    pub clip_from_screen: glm::TMat4<f32>,
    pub sun_shadow_matrices: [glm::TMat4<f32>; CascadedShadowMap::CASCADE_COUNT],
    pub camera_position: glm::TVec4<f32>,
    pub sun_direction: glm::TVec4<f32>,
    pub sun_luminance: [f32; 4],
    pub sun_shadowmap_idx: u32,
    pub time: f32,
    pub stars_threshold: f32, // modifies the number of stars that are visible
	pub stars_exposure: f32,  // modifies the overall strength of the stars
    pub fog_density: f32,
    pub sunzenith_idx: u32,
    pub viewzenith_idx: u32,
    pub sunview_idx: u32,
    pub exposure: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
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
    pub fn framebuffer(&self) -> vk::Framebuffer { self.framebuffer }
    pub fn image(&self) -> vk::Image { self.image }
    pub fn resolution(&self) -> u32 { self.resolution }
    pub fn texture_index(&self) -> u32 { self.texture_index }
    pub fn view(&self) -> vk::ImageView { self.image_view }

    pub fn new(
        vk: &mut VulkanAPI,
        renderer: &mut Renderer,
        resolution: u32,
        clipping_from_view: &glm::TMat4<f32>,
        render_pass: vk::RenderPass
    ) -> Self {
        let format = vk::Format::D32_SFLOAT;

        let image = unsafe {
            let extent = vk::Extent3D {
                width: resolution * Self::CASCADE_COUNT as u32,
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
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };

            let depth_image = vk.device.create_image(&create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            vkutil::allocate_image_memory(vk, depth_image);
            depth_image
        };

        let image_view = unsafe {
            let view_info = vk::ImageViewCreateInfo {
                image,
                format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: vkutil::COMPONENT_MAPPING_DEFAULT,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                },
                ..Default::default()
            };
            vk.device.create_image_view(&view_info, vkutil::MEMORY_ALLOCATOR).unwrap()
        };

        //Create framebuffer
        let framebuffer = unsafe {
            let attachments = [image_view];
            let fb_info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: resolution * Self::CASCADE_COUNT as u32,
                height: resolution,
                layers: 1,
                ..Default::default()
            };
            
            vk.device.create_framebuffer(&fb_info, vkutil::MEMORY_ALLOCATOR).unwrap()
        };

        //Manually picking the cascade distances because math is hard
        //The shadow cascade distances are negative bc they apply to view space
        let near_distance = 1.0;
        let mut view_distances = [0.0; Self::CASCADE_COUNT + 1];
        view_distances[0] = -(near_distance);
        view_distances[1] = -(near_distance + 10.0);
        view_distances[2] = -(near_distance + 40.0);
        view_distances[3] = -(near_distance + 100.0);
        view_distances[4] = -(near_distance + 250.0);
        view_distances[5] = -(near_distance + 500.0);
        view_distances[6] = -(near_distance + 1000.0);

        // let far_distance = 1000.0;
        // let ratio = f32::powf(far_distance / near_distance, 1.0 / Self::CASCADE_COUNT as f32);
        // view_distances[0] = -(near_distance);
        // for i in 1..(Self::CASCADE_COUNT + 1) {
        //     view_distances[i] = ratio * view_distances[i - 1];
        // }
        // println!("{:#?}", view_distances);

        //Compute the clip space distances
        let mut clip_distances = [0.0; Self::CASCADE_COUNT + 1];
        for i in 0..view_distances.len() {
            let p = clipping_from_view * glm::vec4(0.0, 0.0, view_distances[i], 1.0);
            clip_distances[i] = p.z;
        }
        
        let texture_index = renderer.global_textures.insert(vk::DescriptorImageInfo {
            sampler: renderer.point_sampler,
            image_view,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        }) as u32;

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
    pub shadow_map: CascadedShadowMap
}

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub color_format: vk::Format,
    pub depth_format: vk::Format,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_framebuffers: Vec<vk::Framebuffer>
}

impl Swapchain {
    pub fn init(vk: &mut VulkanAPI, render_pass: vk::RenderPass) -> Self {
        //Create the main swapchain for window present
        let vk_swapchain_image_format;
        let vk_swapchain_extent;
        let vk_swapchain = unsafe {
            let present_modes = vk.ext_surface.get_physical_device_surface_present_modes(vk.physical_device, vk.surface).unwrap();
            let surf_capabilities = vk.ext_surface.get_physical_device_surface_capabilities(vk.physical_device, vk.surface).unwrap();
            let surf_formats = vk.ext_surface.get_physical_device_surface_formats(vk.physical_device, vk.surface).unwrap();
            
            //Search for an SRGB swapchain format
            let mut surf_format = vk::SurfaceFormatKHR::default();
            for sformat in surf_formats.iter() {
                if sformat.format == vk::Format::B8G8R8A8_SRGB {
                    surf_format = *sformat;
                    break;
                }
            }

            let present_mode = present_modes[0];
            //let present_mode = vk::PresentModeKHR::MAILBOX;

            vk_swapchain_image_format = surf_format.format;
            vk_swapchain_extent = vk::Extent2D {
                width: surf_capabilities.current_extent.width,
                height: surf_capabilities.current_extent.height
            };
            let create_info = vk::SwapchainCreateInfoKHR {
                surface: vk.surface,
                min_image_count: surf_capabilities.min_image_count,
                image_format: vk_swapchain_image_format,
                image_color_space: surf_format.color_space,
                image_extent: surf_capabilities.current_extent,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 1,
                p_queue_family_indices: [vk.queue_family_index].as_ptr(),
                pre_transform: surf_capabilities.current_transform,
                composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                present_mode,
                ..Default::default()
            };

            let sc = vk.ext_swapchain.create_swapchain(&create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            sc
        };
        
        let vk_swapchain_image_views = unsafe {
            let vk_swapchain_images = vk.ext_swapchain.get_swapchain_images(vk_swapchain).unwrap();

            let mut image_views = Vec::with_capacity(vk_swapchain_images.len());
            for i in 0..vk_swapchain_images.len() {
                let image_subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                };
                let view_info = vk::ImageViewCreateInfo {
                    image: vk_swapchain_images[i],
                    format: vk_swapchain_image_format,
                    view_type: vk::ImageViewType::TYPE_2D,
                    components: vkutil::COMPONENT_MAPPING_DEFAULT,
                    subresource_range: image_subresource_range,
                    ..Default::default()
                };

                image_views.push(vk.device.create_image_view(&view_info, vkutil::MEMORY_ALLOCATOR).unwrap());
            }

            image_views
        };

        let vk_depth_format = vk::Format::D32_SFLOAT;
        let vk_depth_image = unsafe {
            let surf_capabilities = vk.ext_surface.get_physical_device_surface_capabilities(vk.physical_device, vk.surface).unwrap();
            let extent = vk::Extent3D {
                width: surf_capabilities.current_extent.width,
                height: surf_capabilities.current_extent.height,
                depth: 1
            };

            let create_info = vk::ImageCreateInfo {
                queue_family_index_count: 1,
                p_queue_family_indices: [vk.queue_family_index].as_ptr(),
                flags: vk::ImageCreateFlags::empty(),
                image_type: vk::ImageType::TYPE_2D,
                format: vk_depth_format,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let depth_image = vk.device.create_image(&create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            vkutil::allocate_image_memory(vk, depth_image);
            depth_image
        };

        let vk_depth_image_view = unsafe {
            let image_subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
            let view_info = vk::ImageViewCreateInfo {
                image: vk_depth_image,
                format: vk_depth_format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: vkutil::COMPONENT_MAPPING_DEFAULT,
                subresource_range: image_subresource_range,
                ..Default::default()
            };

            vk.device.create_image_view(&view_info, vkutil::MEMORY_ALLOCATOR).unwrap()
        };

        //Create framebuffers
        let swapchain_framebuffers = unsafe {
            let mut attachments = [vk::ImageView::default(), vk_depth_image_view];
            let fb_info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: vk_swapchain_extent.width,
                height: vk_swapchain_extent.height,
                layers: 1,
                ..Default::default()
            };
    
            let mut fbs = Vec::with_capacity(vk_swapchain_image_views.len());
            for view in vk_swapchain_image_views.iter() {
                attachments[0] = view.clone();
                fbs.push(vk.device.create_framebuffer(&fb_info, vkutil::MEMORY_ALLOCATOR).unwrap())
            }
    
            fbs
        };

        Swapchain {
            swapchain: vk_swapchain,
            extent: vk_swapchain_extent,
            color_format: vk_swapchain_image_format,
            depth_format: vk_depth_format,
            depth_image: vk_depth_image,
            swapchain_image_views: vk_swapchain_image_views,
            depth_image_view: vk_depth_image_view,
            swapchain_framebuffers
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct FrameBuffer {
    pub framebuffer_object: vk::Framebuffer,
    pub color_buffer: vk::Image,
    pub color_buffer_view: vk::ImageView,
    pub depth_buffer: vk::Image,
    pub depth_buffer_view: vk::ImageView,
    pub texture_index: u32
}

#[derive(Clone, Copy)]
pub struct InFlightFrameData {
    pub main_command_buffer: vk::CommandBuffer,
    pub swapchain_command_buffer: vk::CommandBuffer,
    pub semaphore: vk::Semaphore,
    pub fence: vk::Fence,
    pub framebuffer: FrameBuffer,
    pub instance_data_start_offset: u64,
    pub instance_data_size: u64
}

pub struct Renderer {
    pub default_diffuse_idx: u32,
    pub default_normal_idx: u32,
    pub default_metal_roughness_idx: u32,
    pub default_emissive_idx: u32,

    pub material_sampler: vk::Sampler,
    pub point_sampler: vk::Sampler,

    primitives: OptionVec<Primitive>,
    drawlist: Vec<DrawCall>,
    instance_data: Vec<InstanceData>,

    pub swapchain: Swapchain,
    
    pub main_sun: Option<SunLight>,
    pub uniform_data: FrameUniforms,

    //Various GPU allocated buffers
    pub position_buffer: GPUBuffer,
    position_offset: u64,
    pub tangent_buffer: GPUBuffer,
    tangent_offset: u64,
    pub normal_buffer: GPUBuffer,
    normal_offset: u64,
    pub uv_buffer: GPUBuffer,
    uv_offset: u64,
    pub imgui_buffer: GPUBuffer,
    pub uniform_buffer: GPUBuffer,
    pub instance_buffer: GPUBuffer,
    pub material_buffer: GPUBuffer,

    pub global_textures: FreeList<DescriptorImageInfo>,
    pub default_texture_idx: u32,
    pub global_materials: FreeList<Material>,

    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub samplers_descriptor_index: u32,
    frames_in_flight: Vec<InFlightFrameData>,
    in_flight_frame: usize
}

impl Renderer {
    pub const FRAMES_IN_FLIGHT: usize = 2;

    pub fn current_in_flight_frame(&self) -> usize { self.in_flight_frame }

    pub fn in_flight_fences(&self) -> [vk::Fence; Self::FRAMES_IN_FLIGHT] {
        let mut fences = [vk::Fence::default(); Self::FRAMES_IN_FLIGHT];
        for i in 0..self.frames_in_flight.len() {
            fences[i] = self.frames_in_flight[i].fence;
        }
        fences
    }

    pub fn framebuffers(&self) -> [FrameBuffer; Self::FRAMES_IN_FLIGHT] {
        let mut fs = [FrameBuffer::default(); Self::FRAMES_IN_FLIGHT];
        for i in 0..self.frames_in_flight.len() {
            fs[i] = self.frames_in_flight[i].framebuffer;
        }
        fs
    }

    pub fn get_instance_data(&self) -> &Vec<InstanceData> {
        &self.instance_data
    }

    pub unsafe fn cleanup(&mut self, vk: &mut VulkanAPI) {
        vk.device.wait_for_fences(&self.in_flight_fences(), true, vk::DeviceSize::MAX).unwrap();
    }

    pub fn init(vk: &mut VulkanAPI, swapchain_render_pass: vk::RenderPass, hdr_render_pass: vk::RenderPass) -> Self {
        //Maintain free list for texture allocation
        let mut global_textures = FreeList::with_capacity(1024);

        //Allocate buffer for frame-constant uniforms
        let uniform_buffer_alignment = vk.physical_device_properties.limits.min_uniform_buffer_offset_alignment;
        let uniform_buffer_size = Self::FRAMES_IN_FLIGHT as u64 * size_to_alignment!(size_of::<FrameUniforms>() as vk::DeviceSize, uniform_buffer_alignment);
        let uniform_buffer = GPUBuffer::allocate(
            vk,
            uniform_buffer_size,
            uniform_buffer_alignment,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu
        );
        
        let storage_buffer_alignment = vk.physical_device_properties.limits.min_storage_buffer_offset_alignment;
        
        //Allocate buffer for instance data
        let max_instances = 1024 * 4;
        let buffer_size = (size_of::<render::InstanceData>() * max_instances * Self::FRAMES_IN_FLIGHT) as vk::DeviceSize;
        let instance_buffer = GPUBuffer::allocate(
            vk,
            buffer_size,
            storage_buffer_alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu
        );

        //Allocate material buffer
        let global_material_slots = 1024 * 4;
        let buffer_size = (global_material_slots * size_of::<MaterialData>()) as vk::DeviceSize;
        let material_buffer = GPUBuffer::allocate(
            vk,
            buffer_size,
            storage_buffer_alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly
        );

        let max_vertices = 1024 * 1024 * 16;
        let alignment = vk.physical_device_properties.limits.min_storage_buffer_offset_alignment;
        let usage_flags = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

        //Allocate position buffer
        let position_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec4<f32>>() as u64,
            alignment,
            usage_flags,
            MemoryLocation::GpuOnly
        );

        //Allocate tangent buffer
        let tangent_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec4<f32>>() as u64,
            alignment,
            usage_flags,
            MemoryLocation::GpuOnly
        );

        //Allocate normal buffer
        let normal_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec4<f32>>() as u64,
            alignment,
            usage_flags,
            MemoryLocation::GpuOnly
        );

        //Allocate uv buffer
        let uv_buffer = GPUBuffer::allocate(
            vk,
            max_vertices * size_of::<glm::TVec2<f32>>() as u64,
            alignment,
            usage_flags,
            MemoryLocation::GpuOnly
        );

        //Allocate imgui buffer
        let max_imgui_vertices = 1024 * 1024;
        let imgui_buffer = GPUBuffer::allocate(
            vk,
            DevGui::FLOATS_PER_VERTEX as u64 * max_imgui_vertices * size_of::<f32>() as u64,
            alignment,
            usage_flags,
            MemoryLocation::CpuToGpu
        );

        //Set up global bindless descriptor set
        let descriptor_set_layout;
        let samplers_descriptor_index;
        let bindless_descriptor_set = unsafe {
            struct BufferDescriptorDesc {
                ty: vk::DescriptorType,
                stage_flags: vk::ShaderStageFlags,
                count: u32,
                buffer: vk::Buffer,
                offset: u64,
                length: u64
            }

            //Bindless descriptor set specification
            let buffer_descriptor_descs = [
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    count: 1,
                    buffer: uniform_buffer.backing_buffer(),
                    offset: 0,
                    length: size_of::<FrameUniforms>() as vk::DeviceSize
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: instance_buffer.backing_buffer(),
                    offset: 0,
                    length: (size_of::<InstanceData>() * max_instances) as vk::DeviceSize
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    count: 1,
                    buffer: material_buffer.backing_buffer(),
                    offset: 0,
                    length: material_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: position_buffer.backing_buffer(),
                    offset: 0,
                    length: position_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: tangent_buffer.backing_buffer(),
                    offset: 0,
                    length: tangent_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: normal_buffer.backing_buffer(),
                    offset: 0,
                    length: normal_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: uv_buffer.backing_buffer(),
                    offset: 0,
                    length: uv_buffer.length()
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    count: 1,
                    buffer: imgui_buffer.backing_buffer(),
                    offset: 0,
                    length: imgui_buffer.length()
                }
            ];
            
            let mut bindings = Vec::new();
            let mut pool_sizes = Vec::new();

            for i in 0..buffer_descriptor_descs.len() {
                let desc = &buffer_descriptor_descs[i];
                let binding = vk::DescriptorSetLayoutBinding {
                    binding: i as u32,
                    descriptor_type: desc.ty,
                    descriptor_count: desc.count,
                    stage_flags: desc.stage_flags,
                    ..Default::default()
                };
                bindings.push(binding);
                let pool_size = vk::DescriptorPoolSize {
                    ty: desc.ty,
                    descriptor_count: 1
                };
                pool_sizes.push(pool_size);
            }

            //Add global texture descriptor
            let binding = vk::DescriptorSetLayoutBinding {
                binding: buffer_descriptor_descs.len() as u32,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: global_textures.size() as u32,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            };
            bindings.push(binding);
            let pool_size = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: global_textures.size() as u32
            };
            pool_sizes.push(pool_size);

            samplers_descriptor_index = pool_sizes.len() as u32 - 1;

            let total_set_count = 1;
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
                flags: vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,    //Allows descriptor sets to be updated even after they're bound
                max_sets: total_set_count,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            };
            let descriptor_pool = vk.device.create_descriptor_pool(&descriptor_pool_info, vkutil::MEMORY_ALLOCATOR).unwrap();

            let mut flag_list = vec![vk::DescriptorBindingFlags::default(); bindings.len()];
            flag_list[samplers_descriptor_index as usize] = 
                vk::DescriptorBindingFlags::PARTIALLY_BOUND;
            
            let binding_info = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT {
                binding_count: flag_list.len() as u32,
                p_binding_flags: flag_list.as_ptr(),
                ..Default::default()
            };
            let descriptor_layout = vk::DescriptorSetLayoutCreateInfo {
                p_next: &binding_info as *const _ as *const c_void,
                binding_count: bindings.len() as u32,
                p_bindings: bindings.as_ptr(),
                ..Default::default()
            };

            descriptor_set_layout = vk.device.create_descriptor_set_layout(&descriptor_layout, vkutil::MEMORY_ALLOCATOR).unwrap();

            let vk_alloc_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: total_set_count,
                p_set_layouts: &descriptor_set_layout,
                ..Default::default()
            };
            let descriptor_sets = vk.device.allocate_descriptor_sets(&vk_alloc_info).unwrap();

            for i in 0..buffer_descriptor_descs.len() {
                let desc = &buffer_descriptor_descs[i];
                let info = vk::DescriptorBufferInfo {
                    buffer: desc.buffer,
                    offset: desc.offset,
                    range: desc.length
                };
                let write = vk::WriteDescriptorSet {
                    dst_set: descriptor_sets[0],
                    descriptor_count: 1,
                    descriptor_type: desc.ty,
                    p_buffer_info: &info,
                    dst_array_element: 0,
                    dst_binding: i as u32,
                    ..Default::default()
                };
                vk.device.update_descriptor_sets(&[write], &[]);
            }

            descriptor_sets[0]
        };

        //Create texture samplers
        let (material_sampler, point_sampler) = unsafe {
            let sampler_info = vk::SamplerCreateInfo {
                min_filter: vk::Filter::LINEAR,
                mag_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                address_mode_w: vk::SamplerAddressMode::REPEAT,
                mip_lod_bias: 0.0,
                anisotropy_enable: vk::TRUE,
                max_anisotropy: 16.0,
                compare_enable: vk::FALSE,
                min_lod: 0.0,
                max_lod: vk::LOD_CLAMP_NONE,
                border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
                unnormalized_coordinates: vk::FALSE,
                ..Default::default()
            };
            let mat = vk.device.create_sampler(&sampler_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            
            let sampler_info = vk::SamplerCreateInfo {
                min_filter: vk::Filter::NEAREST,
                mag_filter: vk::Filter::NEAREST,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                anisotropy_enable: vk::FALSE,
                ..sampler_info
            };
            let font = vk.device.create_sampler(&sampler_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            
            (mat, font)
        };

        let default_color_idx = unsafe { global_textures.insert(vkutil::upload_raw_image(vk, point_sampler, vk::Format::R8G8B8A8_UNORM, 1, 1, &[0xFF, 0xFF, 0xFF, 0xFF])) as u32};
        let default_metalrough_idx = unsafe { global_textures.insert(vkutil::upload_raw_image(vk, point_sampler, vk::Format::R8G8B8A8_UNORM, 1, 1, &[0xFF, 0xFF, 0x00, 0xFF])) as u32};
        let default_emissive_idx = unsafe { global_textures.insert(vkutil::upload_raw_image(vk, point_sampler, vk::Format::R8G8B8A8_UNORM, 1, 1, &[0x00, 0x00, 0x00, 0xFF])) as u32};
        let default_normal_idx = unsafe { global_textures.insert(vkutil::upload_raw_image(vk, point_sampler, vk::Format::R8G8B8A8_UNORM, 1, 1, &[0x80, 0x80, 0xFF, 0xFF])) as u32};

        //Create free list for materials
        let global_materials = FreeList::with_capacity(256);

        let mut uniforms = FrameUniforms::default();
        uniforms.exposure = 1.0;

        //Load environment textures
        {
            let sunzenith_index = vkutil::load_bc7_texture(vk, &mut global_textures, material_sampler, "./data/textures/sunzenith_gradient.dds", ColorSpace::SRGB);
            let viewzenith_index = vkutil::load_bc7_texture(vk, &mut global_textures, material_sampler, "./data/textures/viewzenith_gradient.dds", ColorSpace::SRGB);
            let sunview_index = vkutil::load_bc7_texture(vk, &mut global_textures, material_sampler, "./data/textures/sunview_gradient.dds", ColorSpace::SRGB);
            
            uniforms.sunzenith_idx = sunzenith_index;
            uniforms.viewzenith_idx = viewzenith_index;
            uniforms.sunview_idx = sunview_index;
        };

        //Create the main swapchain for window present
        let swapchain = Swapchain::init(vk, swapchain_render_pass);
        
        let surf_capabilities = unsafe { vk.ext_surface.get_physical_device_surface_capabilities(vk.physical_device, vk.surface).unwrap() };
        let primary_framebuffer_extent = vk::Extent3D {
            width: surf_capabilities.current_extent.width,
            height: surf_capabilities.current_extent.height,
            depth: 1
        };

        //Create main depth buffer
        let vk_depth_format = vk::Format::D32_SFLOAT;
        let depth_buffer_image = unsafe {
            let create_info = vk::ImageCreateInfo {
                queue_family_index_count: 1,
                p_queue_family_indices: [vk.queue_family_index].as_ptr(),
                flags: vk::ImageCreateFlags::empty(),
                image_type: vk::ImageType::TYPE_2D,
                format: vk_depth_format,
                extent: primary_framebuffer_extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let depth_image = vk.device.create_image(&create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            vkutil::allocate_image_memory(vk, depth_image);
            depth_image
        };

        let depth_buffer_view = unsafe {
            let image_subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            };
            let view_info = vk::ImageViewCreateInfo {
                image: depth_buffer_image,
                format: vk_depth_format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: vkutil::COMPONENT_MAPPING_DEFAULT,
                subresource_range: image_subresource_range,
                ..Default::default()
            };

            vk.device.create_image_view(&view_info, vkutil::MEMORY_ALLOCATOR).unwrap()
        };

        //Initialize per-frame rendering state
        let command_buffers = {
            let hdr_color_format = vk::Format::R32G32B32A32_SFLOAT;

            let mut color_buffers = [vk::Image::default(); Self::FRAMES_IN_FLIGHT];
            let mut color_buffer_views = [vk::ImageView::default(); Self::FRAMES_IN_FLIGHT];
            let mut hdr_framebuffers = [vk::Framebuffer::default(); Self::FRAMES_IN_FLIGHT];
            for i in 0..Self::FRAMES_IN_FLIGHT {
                let primary_color_buffer = unsafe {
                    let create_info = vk::ImageCreateInfo {
                        queue_family_index_count: 1,
                        p_queue_family_indices: [vk.queue_family_index].as_ptr(),
                        flags: vk::ImageCreateFlags::empty(),
                        image_type: vk::ImageType::TYPE_2D,
                        format: hdr_color_format,
                        extent: primary_framebuffer_extent,
                        mip_levels: 1,
                        array_layers: 1,
                        samples: vk::SampleCountFlags::TYPE_1,
                        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        ..Default::default()
                    };

                    let image = vk.device.create_image(&create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
                    vkutil::allocate_image_memory(vk, image);
                    image
                };
                color_buffers[i] = primary_color_buffer;

                let color_buffer_view = unsafe {
                    let image_subresource_range = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1
                    };
                    let view_info = vk::ImageViewCreateInfo {
                        image: primary_color_buffer,
                        format: hdr_color_format,
                        view_type: vk::ImageViewType::TYPE_2D,
                        components: vkutil::COMPONENT_MAPPING_DEFAULT,
                        subresource_range: image_subresource_range,
                        ..Default::default()
                    };
    
                    vk.device.create_image_view(&view_info, vkutil::MEMORY_ALLOCATOR).unwrap()
                };
                color_buffer_views[i] = color_buffer_view;
            
                //Create framebuffer
                let framebuffer_object = unsafe {
                    let attachments = [color_buffer_view, depth_buffer_view];
                    let fb_info = vk::FramebufferCreateInfo {
                        render_pass: hdr_render_pass,
                        attachment_count: attachments.len() as u32,
                        p_attachments: attachments.as_ptr(),
                        width: primary_framebuffer_extent.width,
                        height: primary_framebuffer_extent.height,
                        layers: 1,
                        ..Default::default()
                    };
                    vk.device.create_framebuffer(&fb_info, vkutil::MEMORY_ALLOCATOR).unwrap()
                };
                hdr_framebuffers[i] = framebuffer_object;
                
            }

            //Data for each in-flight frame
            let command_buffers = {
                let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
                    command_pool: vk.command_pool,
                    command_buffer_count: 2 * Self::FRAMES_IN_FLIGHT as u32,
                    level: vk::CommandBufferLevel::PRIMARY,
                    ..Default::default()
                };
                let command_buffers = unsafe { vk.device.allocate_command_buffers(&command_buffer_alloc_info).unwrap() };
                
                let mut c_buffer_datas = Vec::with_capacity(Self::FRAMES_IN_FLIGHT);
                for i in 0..Self::FRAMES_IN_FLIGHT {
                    let create_info = vk::FenceCreateInfo {
                        flags: vk::FenceCreateFlags::SIGNALED,
                        ..Default::default()
                    };
                    let fence = unsafe { vk.device.create_fence(&create_info, vkutil::MEMORY_ALLOCATOR).unwrap() };
                    let semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap() };
                    let swapchain_semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap() };

                    let descriptor_info = vk::DescriptorImageInfo {
                        sampler: material_sampler,
                        image_view: color_buffer_views[i],
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    };
                    let texture_index = global_textures.insert(descriptor_info) as u32;

                    let framebuffer = FrameBuffer {
                        framebuffer_object: hdr_framebuffers[i],
                        color_buffer: color_buffers[i],
                        color_buffer_view: color_buffer_views[i],
                        depth_buffer: depth_buffer_image,
                        depth_buffer_view,
                        texture_index
                    };
                    
                    let data = InFlightFrameData {
                        main_command_buffer: command_buffers[2 * i],
                        semaphore,
                        swapchain_command_buffer: command_buffers[2 * i + 1],
                        fence,
                        framebuffer,
                        instance_data_start_offset: 0,
                        instance_data_size: 0
                    };
                    c_buffer_datas.push(data);
                }
                c_buffer_datas
            };
            
            command_buffers
        };

        Renderer {
            default_diffuse_idx: default_color_idx,
            default_normal_idx,
            default_metal_roughness_idx: default_metalrough_idx,
            default_emissive_idx,
            material_sampler,
            point_sampler,
            primitives: OptionVec::new(),
            drawlist: Vec::new(),
            instance_data: Vec::new(),
            swapchain,
            uniform_data: uniforms,
            global_textures,
            default_texture_idx: 0,
            global_materials,
            descriptor_set_layout,
            bindless_descriptor_set,
            position_buffer,
            position_offset: 0,
            tangent_buffer,
            tangent_offset: 0,
            normal_buffer,
            normal_offset: 0,
            uv_buffer,
            uv_offset: 0,
            imgui_buffer,
            uniform_buffer,
            instance_buffer,
            material_buffer,
            samplers_descriptor_index,
            main_sun: None,
            frames_in_flight: command_buffers,
            in_flight_frame: 0
        }
    }

    fn next_frame(&mut self, vk: &mut VulkanAPI) -> InFlightFrameData {
        let cb = self.frames_in_flight[self.in_flight_frame];
        unsafe { vk.device.wait_for_fences(&[cb.fence], true, vk::DeviceSize::MAX).unwrap(); }
        self.in_flight_frame += 1;
        self.in_flight_frame %= Self::FRAMES_IN_FLIGHT;
        cb
    }

    pub fn register_model(&mut self, data: Primitive) -> usize {
        self.primitives.insert(data)
    }

    fn upload_vertex_attribute(vk: &mut VulkanAPI, data: &[f32], buffer: &GPUBuffer, offset: &mut u64) -> u32 {
        let old_offset = *offset;
        let new_offset = old_offset + data.len() as u64;
        buffer.upload_subbuffer_elements(vk, data, old_offset);
        *offset = new_offset;
        old_offset.try_into().unwrap()
    }
    
    pub fn append_vertex_positions(&mut self, vk: &mut VulkanAPI, positions: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, positions, &self.position_buffer, &mut self.position_offset) / 4
    }
    
    pub fn append_vertex_tangents(&mut self, vk: &mut VulkanAPI, tangents: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, tangents, &self.tangent_buffer, &mut self.tangent_offset) / 4
    }
    
    pub fn append_vertex_normals(&mut self, vk: &mut VulkanAPI, normals: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, normals, &self.normal_buffer, &mut self.normal_offset) / 4
    }
    
    pub fn append_vertex_uvs(&mut self, vk: &mut VulkanAPI, uvs: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, uvs, &self.uv_buffer, &mut self.uv_offset) / 2
    }

    pub fn replace_vertex_positions(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * 4;
        Self::upload_vertex_attribute(vk, data, &self.position_buffer, &mut my_offset);
    }

    pub fn replace_vertex_tangents(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * 4;
        Self::upload_vertex_attribute(vk, data, &self.tangent_buffer, &mut my_offset);
    }

    pub fn replace_vertex_normals(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * 4;
        Self::upload_vertex_attribute(vk, data, &self.normal_buffer, &mut my_offset);
    }

    pub fn replace_vertex_uvs(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * 2;
        Self::upload_vertex_attribute(vk, data, &self.uv_buffer, &mut my_offset);
    }

    pub fn replace_imgui_vertices(&mut self, vk: &mut VulkanAPI, data: &[f32], offset: u64) {
        let mut my_offset = offset * DevGui::FLOATS_PER_VERTEX as u64;
        Self::upload_vertex_attribute(vk, data, &self.imgui_buffer, &mut my_offset);
    }

    pub fn get_model(&self, idx: usize) -> &Option<Primitive> {
        &self.primitives[idx]
    }

    pub fn prepare_frame(&mut self, vk: &mut VulkanAPI, window_size: glm::TVec2<u32>, view_from_world: &glm::TMat4<f32>, elapsed_time: f32) -> InFlightFrameData {
        //Wait for LRU frame to finish
        let frame_info = self.next_frame(vk);

        //Update bindless texture sampler descriptors
        if self.global_textures.updated {
            self.global_textures.updated = false;

            let mut image_infos = vec![self.global_textures[self.default_texture_idx.try_into().unwrap()].unwrap(); self.global_textures.size() as usize];
            for i in 0..self.global_textures.len() {
                if let Some(info) = self.global_textures[i] {
                    image_infos[i] = info;
                }
            }

            let sampler_write = vk::WriteDescriptorSet {
                dst_set: self.bindless_descriptor_set,
                descriptor_count: self.global_textures.size() as u32,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: image_infos.as_ptr(),
                dst_array_element: 0,
                dst_binding: self.samplers_descriptor_index,
                ..Default::default()
            };
            unsafe { vk.device.update_descriptor_sets(&[sampler_write], &[]); }
        }

        //Update bindless material definitions
        if self.global_materials.updated {
            self.global_materials.updated = false;

            let mut upload_mats = Vec::with_capacity(self.global_materials.len());
            for i in 0..self.global_materials.len() {
                if let Some(mat) = &self.global_materials[i] {
                    upload_mats.push(mat.data());
                }
            }

            self.material_buffer.upload_buffer(vk, &upload_mats);
        }
        
        //Update uniform/storage buffers
        {
            let uniforms = &mut self.uniform_data;
            //Update static scene data
            uniforms.clip_from_screen = glm::mat4(
                2.0 / window_size.x as f32, 0.0, 0.0, -1.0,
                0.0, 2.0 / window_size.y as f32, 0.0, -1.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            );

            let near_distance = 0.1;
            let far_distance = 1000.0;
            let projection_matrix = glm::perspective_fov_rh_zo(glm::half_pi::<f32>(), window_size.x as f32, window_size.y as f32, near_distance, far_distance);
            uniforms.clip_from_view = glm::mat4(
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ) * projection_matrix;
    
            uniforms.clip_from_world = uniforms.clip_from_view * view_from_world;

            if let Some(sunlight) = &self.main_sun {
                //Compute sun direction from pitch and yaw
                uniforms.sun_direction = 
                    glm::rotation(sunlight.yaw, &glm::vec3(0.0, 0.0, 1.0)) *
                    glm::rotation(sunlight.pitch, &glm::vec3(0.0, 1.0, 0.0)) *
                    glm::vec4(-1.0, 0.0, 0.0, 0.0);

                uniforms.sun_shadow_matrices = sunlight.shadow_map.compute_shadow_cascade_matrices(
                    &uniforms.sun_direction.xyz(),
                    &uniforms.view_from_world,
                    &uniforms.clip_from_view
                );

                uniforms.sun_shadow_distances = sunlight.shadow_map.clip_distances().clone();
            }
            
            //Compute the view-projection matrix for the skybox (the conversion functions are just there to nullify the translation component of the view matrix)
            //The skybox vertices should be rotated along with the camera, but they shouldn't be translated in order to maintain the illusion
            //that the sky is infinitely far away
            uniforms.clip_from_skybox = uniforms.clip_from_view * glm::mat3_to_mat4(&glm::mat4_to_mat3(&view_from_world));

            uniforms.time = elapsed_time;

            let dynamic_offset = (self.in_flight_frame as u64 * size_to_alignment!(size_of::<render::FrameUniforms>() as u64, vk.physical_device_properties.limits.min_uniform_buffer_offset_alignment)) as u64;
            
            let uniform_bytes = struct_to_bytes(&self.uniform_data);
            self.uniform_buffer.upload_subbuffer_elements(vk, uniform_bytes, dynamic_offset);
        };

        //Update instance data storage buffer
        {
            let instance_data_bytes = slice_to_bytes(&self.instance_data);
            let last_frame_data = &mut self.frames_in_flight[self.in_flight_frame.overflowing_sub(Self::FRAMES_IN_FLIGHT - 1).0 % Self::FRAMES_IN_FLIGHT];
            let start_of_first_live_data = last_frame_data.instance_data_start_offset;
            let start_offset = if start_of_first_live_data > instance_data_bytes.len() as u64 {
                0
            } else {
                last_frame_data.instance_data_start_offset + last_frame_data.instance_data_size
            };
            let start_offset = size_to_alignment!(start_offset, vk.physical_device_properties.limits.min_storage_buffer_offset_alignment);

            self.frames_in_flight[self.in_flight_frame].instance_data_start_offset = start_offset;
            self.instance_buffer.upload_subbuffer_bytes(vk, instance_data_bytes, start_offset);
        }

        frame_info
    }

    pub fn queue_drawcall(&mut self, model_idx: usize, transforms: &[glm::TMat4<f32>]) {
        match &self.primitives[model_idx] {
            None => {
                tfd::message_box_ok("No model at supplied index", &format!("No model loaded at index {}", model_idx), tfd::MessageBoxIcon::Error);
                return;
            }
            Some(prim) => {let instance_count = transforms.len() as u32;
                let first_instance = self.instance_data.len() as u32;
        
                let pipeline = self.global_materials[prim.material_idx.try_into().unwrap()].as_ref().unwrap().pipeline;
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
        }
    }

    pub fn drawlist_iter(&self) -> Iter<DrawCall> {
        self.drawlist.iter()
    }

    pub fn reset(&mut self) {
        self.drawlist.clear();
        self.instance_data.clear();
    }
}
