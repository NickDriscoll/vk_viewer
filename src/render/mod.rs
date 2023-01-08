#![allow(unused)]

use core::slice::Iter;
use std::{convert::TryInto, ffi::c_void, hash::{Hash, Hasher}, collections::hash_map::DefaultHasher};
use ozy::io::OzyMesh;
use slotmap::{SlotMap, new_key_type};
use crate::*;
pub use structs::*;
use vkdevice::*;

pub mod structs;
pub mod vkdevice;

pub struct WindowManager {
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub color_format: vk::Format,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_framebuffers: Vec<vk::Framebuffer>
}

impl WindowManager {
    pub fn init(vk: &mut VulkanGraphicsDevice, sdl_window: &sdl2::video::Window, render_pass: vk::RenderPass) -> Self {
        //Use SDL to create the Vulkan surface
        let vk_surface = {
            use ash::vk::Handle;
            let raw_surf = sdl_window.vulkan_create_surface(vk.instance.handle().as_raw() as usize).unwrap();
            vk::SurfaceKHR::from_raw(raw_surf)
        };
    
        //Check that we can do swapchain present on this window
        unsafe {
            if !vk.ext_surface.get_physical_device_surface_support(vk.physical_device, vk.queue_family_index, vk_surface).unwrap() {
                crash_with_error_dialog("Swapchain present is unavailable on the selected device queue.\nThe application will now exit.");
            }
        }

        //Create the main swapchain for window present
        let vk_swapchain_image_format;
        let vk_swapchain_extent;
        let vk_swapchain = unsafe {
            let present_modes = vk.ext_surface.get_physical_device_surface_present_modes(vk.physical_device, vk_surface).unwrap();
            let surf_capabilities = vk.ext_surface.get_physical_device_surface_capabilities(vk.physical_device, vk_surface).unwrap();
            let surf_formats = vk.ext_surface.get_physical_device_surface_formats(vk.physical_device, vk_surface).unwrap();
            
            //Search for an SRGB swapchain format
            let mut surf_format = vk::SurfaceFormatKHR::default();
            for sformat in surf_formats.iter() {
                if sformat.format == vk::Format::B8G8R8A8_SRGB {
                    surf_format = *sformat;
                    break;
                }
            }

            let desired_present_mode = vk::PresentModeKHR::FIFO;
            let mut has_fifo = false;
            for mode in present_modes {
                if mode == desired_present_mode {
                    has_fifo = true;
                    break;
                }
            }
            if !has_fifo {
                crash_with_error_dialog("FIFO present mode not supported on your system.");
            }
            let present_mode = desired_present_mode;
            //let present_mode = vk::PresentModeKHR::MAILBOX;

            vk_swapchain_image_format = surf_format.format;
            vk_swapchain_extent = vk::Extent2D {
                width: surf_capabilities.current_extent.width,
                height: surf_capabilities.current_extent.height
            };
            let create_info = vk::SwapchainCreateInfoKHR {
                surface: vk_surface,
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

            let sc = vk.ext_swapchain.create_swapchain(&create_info, vkdevice::MEMORY_ALLOCATOR).unwrap();
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
                    components: vkdevice::COMPONENT_MAPPING_DEFAULT,
                    subresource_range: image_subresource_range,
                    ..Default::default()
                };

                image_views.push(vk.device.create_image_view(&view_info, vkdevice::MEMORY_ALLOCATOR).unwrap());
            }

            image_views
        };

        //Create framebuffers
        let swapchain_framebuffers = unsafe {
            let mut attachment = vk::ImageView::default();
            let fb_info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: 1,
                p_attachments: &attachment,
                width: vk_swapchain_extent.width,
                height: vk_swapchain_extent.height,
                layers: 1,
                ..Default::default()
            };
    
            let mut fbs = Vec::with_capacity(vk_swapchain_image_views.len());
            for view in vk_swapchain_image_views.iter() {
                attachment = view.clone();
                fbs.push(vk.device.create_framebuffer(&fb_info, vkdevice::MEMORY_ALLOCATOR).unwrap())
            }
    
            fbs
        };

        WindowManager {
            surface: vk_surface,
            swapchain: vk_swapchain,
            extent: vk_swapchain_extent,
            color_format: vk_swapchain_image_format,
            swapchain_image_views: vk_swapchain_image_views,
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

struct BufferBlock {
    pub start_offset: u64,       //In f32s,
    pub length: u64              //In f32s
}

pub struct InstancedSlotMap<K: slotmap::Key, V: UniqueID> {
    items: SlotMap<K, V>,
    counts: HashMap<K, u32>
}

impl<K: slotmap::Key, V: UniqueID> InstancedSlotMap<K, V> {
    pub fn with_key() -> Self {
        InstancedSlotMap {
            items: SlotMap::with_key(),
            counts: HashMap::new()
        }
    }

    pub fn with_capacity_and_key(capacity: usize) -> Self {
        InstancedSlotMap {
            items: SlotMap::with_capacity_and_key(capacity),
            counts: HashMap::with_capacity(capacity)
        }
    }

    pub fn get(&self, key: K) -> Option<&V> {
        self.items.get(key)
    }

    pub fn id_exists(&self, id: u64) -> Option<K> {
        let mut res = None;
        for item in self.items.iter() {
            if item.1.id() == id {
                res = Some(item.0);
                break;
            }
        }
        res
    }

    //This is to insert a brand new value
    pub fn insert(&mut self, value: V) -> K {
        let key = self.items.insert(value);
        self.counts.insert(key, 1);
        key
    }

    pub fn increment_model_count(&mut self, key: K) {
        if let Some(count) = self.counts.get_mut(&key) {
            *count += 1;
        }
    }

    //Returns the deleted value if the last instance was deleted
    pub fn delete_instance(&mut self, key: K) -> Option<V> {
        let mut last_one = None;
        if let Some(count) = self.counts.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                self.counts.remove(&key);
                last_one = self.items.remove(key);
            }
        }

        last_one
    }
}

pub struct Renderer {
    pub default_color_idx: u32,
    pub default_normal_idx: u32,
    pub default_metal_roughness_idx: u32,
    pub default_emissive_idx: u32,
    pub default_texture_idx: u32,

    pub material_sampler: vk::Sampler,
    pub point_sampler: vk::Sampler,
    pub shadow_sampler: vk::Sampler,

    models: InstancedSlotMap<ModelKey, Model>,
    primitives: SlotMap<PrimitiveKey, Primitive>,
    delete_queue: Vec<DeferredDelete>,
    instance_data: Vec<InstanceData>,
    raw_draws: Vec<DesiredDraw>,
    drawstream: Vec<DrawCall>,

    pub window_manager: WindowManager,
    
    //Light data
    directional_lights: SlotMap<DirectionalLightKey, SunLight>,
    pub irradiance_map_idx: u32,

    pub uniform_data: EnvironmentUniforms,

    //Various GPU allocated buffers
    pub position_buffer: GPUBuffer,
    position_buffer_blocks: SlotMap<PositionBufferBlockKey, BufferBlock>,
    position_offset: u64,               //In f32s
    pub tangent_buffer: GPUBuffer,
    tangent_offset: u64,                //In f32s
    pub normal_buffer: GPUBuffer,
    normal_offset: u64,                 //In f32s
    pub uv_buffer: GPUBuffer,
    uv_offset: u64,                     //In f32s
    pub imgui_buffer: GPUBuffer,
    pub uniform_buffer: GPUBuffer,
    pub instance_buffer: GPUBuffer,
    pub material_buffer: GPUBuffer,
    
    pub compute_buffer: GPUBuffer,

    pub global_images: FreeList<GPUImage>,
    pub global_materials: FreeList<Material>,

    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub samplers_descriptor_index: u32,
    frames_in_flight: Vec<InFlightFrameData>,
    in_flight_frame: usize
}

impl Renderer {
    pub const NEAR_CLIP_DISTANCE: f32 = 0.1;
    pub const FAR_CLIP_DISTANCE: f32 = 1000.0;
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

    pub fn get_model(&self, key: ModelKey) -> Option<&Model> {
        self.models.get(key)
    }

    pub unsafe fn cleanup(&mut self, vk: &mut VulkanGraphicsDevice) {
        vk.device.wait_for_fences(&self.in_flight_fences(), true, vk::DeviceSize::MAX).unwrap();
    }

    pub fn increment_model_count(&mut self, key: ModelKey) {
        if let Some(count) = self.models.counts.get_mut(&key) {
            *count += 1;
        }
    }

    pub fn init(vk: &mut VulkanGraphicsDevice, window: &sdl2::video::Window, swapchain_render_pass: vk::RenderPass, hdr_render_pass: vk::RenderPass) -> Self {
        //Allocate buffer for frame-constant uniforms
        let uniform_buffer_alignment = vk.physical_device_properties.limits.min_uniform_buffer_offset_alignment;
        let uniform_buffer_size = Self::FRAMES_IN_FLIGHT as u64 * size_to_alignment!(size_of::<EnvironmentUniforms>() as vk::DeviceSize, uniform_buffer_alignment);
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
        
        let compute_buffer = GPUBuffer::allocate(
            vk,
            (3840 * 2160 * size_of::<u32>()) as u64,
            alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly
        );
            
        //Maintain free list for texture allocation
        let mut global_images = FreeList::with_capacity(1024);

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
                    length: size_of::<EnvironmentUniforms>() as vk::DeviceSize
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
                },
                BufferDescriptorDesc {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    count: 1,
                    buffer: compute_buffer.backing_buffer(),
                    offset: 0,
                    length: compute_buffer.length()
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
                descriptor_count: global_images.size() as u32,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            };
            bindings.push(binding);
            let pool_size = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: global_images.size() as u32
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
            let descriptor_pool = vk.device.create_descriptor_pool(&descriptor_pool_info, vkdevice::MEMORY_ALLOCATOR).unwrap();

            let mut flag_list = vec![vk::DescriptorBindingFlags::default(); bindings.len()];
            flag_list[samplers_descriptor_index as usize] = vk::DescriptorBindingFlags::PARTIALLY_BOUND;
            
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

            descriptor_set_layout = vk.device.create_descriptor_set_layout(&descriptor_layout, vkdevice::MEMORY_ALLOCATOR).unwrap();

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
        let (material_sampler, point_sampler, shadow_sampler, cubemap_sampler) = unsafe {
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
            let mat = vk.device.create_sampler(&sampler_info, vkdevice::MEMORY_ALLOCATOR).unwrap();
            
            let sampler_info = vk::SamplerCreateInfo {
                min_filter: vk::Filter::NEAREST,
                mag_filter: vk::Filter::NEAREST,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                anisotropy_enable: vk::FALSE,
                ..sampler_info
            };
            let font = vk.device.create_sampler(&sampler_info, vkdevice::MEMORY_ALLOCATOR).unwrap();

            let sampler_info = vk::SamplerCreateInfo {
                min_filter: vk::Filter::LINEAR,
                mag_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                address_mode_w: vk::SamplerAddressMode::REPEAT,
                mip_lod_bias: 0.0,
                compare_enable: vk::TRUE,
                compare_op: vk::CompareOp::LESS_OR_EQUAL,
                min_lod: 0.0,
                max_lod: vk::LOD_CLAMP_NONE,
                border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
                unnormalized_coordinates: vk::FALSE,
                ..Default::default()
            };
            let shadow = vk.device.create_sampler(&sampler_info, vkdevice::MEMORY_ALLOCATOR).unwrap();

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
            let cubemap = vk.device.create_sampler(&sampler_info, vkdevice::MEMORY_ALLOCATOR).unwrap();

            (mat, font, shadow, cubemap)
        };

        let format = vk::Format::R8G8B8A8_UNORM;
        let layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let default_color_idx = unsafe { global_images.insert(vkdevice::upload_raw_image(vk, point_sampler, format, layout, 1, 1, &[0xFF, 0xFF, 0xFF, 0xFF])) as u32};
        let default_metalrough_idx = unsafe { global_images.insert(vkdevice::upload_raw_image(vk, point_sampler, format, layout, 1, 1, &[0xFF, 0xFF, 0xFF, 0xFF])) as u32};
        let default_emissive_idx = unsafe { global_images.insert(vkdevice::upload_raw_image(vk, point_sampler, format, layout, 1, 1, &[0x00, 0x00, 0x00, 0xFF])) as u32};
        let default_normal_idx = unsafe { global_images.insert(vkdevice::upload_raw_image(vk, point_sampler, format, layout, 1, 1, &[0x80, 0x80, 0xFF, 0xFF])) as u32};

        //Create free list for materials
        let global_materials = FreeList::with_capacity(256);

        let mut uniforms = EnvironmentUniforms::default();
        uniforms.exposure = 1.0;
        uniforms.real_sky = 1.0;

        //Load environment textures
        {
            let sunzenith_index = vkdevice::load_bc7_texture(vk, &mut global_images, material_sampler, "./data/textures/sunzenith_gradient.dds");
            let viewzenith_index = vkdevice::load_bc7_texture(vk, &mut global_images, material_sampler, "./data/textures/viewzenith_gradient.dds");
            let sunview_index = vkdevice::load_bc7_texture(vk, &mut global_images, material_sampler, "./data/textures/sunview_gradient.dds");
            
            uniforms.sunzenith_idx = sunzenith_index;
            uniforms.viewzenith_idx = viewzenith_index;
            uniforms.sunview_idx = sunview_index;
        }

        //Create the main swapchain for window present
        let window_manager = WindowManager::init(vk, &window, swapchain_render_pass);
        
        let surf_capabilities = unsafe { vk.ext_surface.get_physical_device_surface_capabilities(vk.physical_device, window_manager.surface).unwrap() };
        let primary_framebuffer_extent = vk::Extent3D {
            width: surf_capabilities.current_extent.width,
            height: surf_capabilities.current_extent.height,
            depth: 1
        };

        let framebuffers = Self::create_hdr_framebuffers(vk, primary_framebuffer_extent, hdr_render_pass, material_sampler, &mut global_images);
        
        //Initialize per-frame rendering state
        let in_flight_frame_data = {
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
                    let fence = unsafe { vk.device.create_fence(&create_info, vkdevice::MEMORY_ALLOCATOR).unwrap() };
                    let semaphore = unsafe { vk.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkdevice::MEMORY_ALLOCATOR).unwrap() };
                    
                    let data = InFlightFrameData {
                        main_command_buffer: command_buffers[2 * i],
                        swapchain_command_buffer: command_buffers[2 * i + 1],
                        semaphore,
                        fence,
                        framebuffer: framebuffers[i],
                        instance_data_start_offset: 0,
                        instance_data_size: 0
                    };
                    c_buffer_datas.push(data);
                }
                c_buffer_datas
            };
            
            command_buffers
        };

        let irradiance_map_info = vk::ImageCreateInfo {
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R16G16B16A16_SFLOAT,
            extent: vk::Extent3D {
                width: 512,
                height: 512,
                depth: 1
            },
            mip_levels: ozy::routines::calculate_mipcount(512, 512),
            array_layers: 6,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &vk.queue_family_index,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let mut irradiance_image = GPUImage::allocate(vk, &irradiance_map_info, cubemap_sampler);
        let irradiance_view_info = vk::ImageViewCreateInfo {
            image: irradiance_image.image,
            view_type: vk::ImageViewType::CUBE,
            format: vk::Format::R16G16B16A16_SFLOAT,
            components: vkdevice::COMPONENT_MAPPING_DEFAULT,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6
            },
            ..Default::default()
        };
        irradiance_image.view = unsafe { Some(vk.device.create_image_view(&irradiance_view_info, vkdevice::MEMORY_ALLOCATOR).unwrap()) };
        let irradiance_map_idx = global_images.insert(irradiance_image) as u32;

        Renderer {
            default_color_idx,
            default_normal_idx,
            default_metal_roughness_idx: default_metalrough_idx,
            default_emissive_idx,
            material_sampler,
            point_sampler,
            shadow_sampler,
            models: InstancedSlotMap::with_key(),
            primitives: SlotMap::with_key(),
            delete_queue: Vec::new(),
            raw_draws: Vec::new(),
            drawstream: Vec::new(),
            instance_data: Vec::new(),
            window_manager,
            uniform_data: uniforms,
            global_images,
            default_texture_idx: 0,
            global_materials,
            descriptor_set_layout,
            bindless_descriptor_set,
            position_buffer,
            position_buffer_blocks: SlotMap::with_key(),
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
            compute_buffer,
            samplers_descriptor_index,
            //main_sun: None,
            directional_lights: SlotMap::with_key(),
            irradiance_map_idx,
            frames_in_flight: in_flight_frame_data,
            in_flight_frame: 0
        }
    }

    fn create_hdr_framebuffers(vk: &mut VulkanGraphicsDevice, extent: vk::Extent3D, hdr_render_pass: vk::RenderPass, sampler: vk::Sampler, global_images: &mut FreeList<GPUImage>) -> [FrameBuffer; Self::FRAMES_IN_FLIGHT] {
        let hdr_color_format = vk::Format::R16G16B16A16_SFLOAT;
        let vk_depth_format = vk::Format::D32_SFLOAT;

        //Create main depth buffer
        let depth_buffer_image = unsafe {
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

            let depth_image = vk.device.create_image(&create_info, vkdevice::MEMORY_ALLOCATOR).unwrap();
            vkdevice::allocate_image_memory(vk, depth_image);
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
                components: vkdevice::COMPONENT_MAPPING_DEFAULT,
                subresource_range: image_subresource_range,
                ..Default::default()
            };

            vk.device.create_image_view(&view_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
        };

        let mut color_buffers = [vk::Image::default(); Self::FRAMES_IN_FLIGHT];
        let mut color_buffer_views = [vk::ImageView::default(); Self::FRAMES_IN_FLIGHT];
        let mut hdr_framebuffers = [vk::Framebuffer::default(); Self::FRAMES_IN_FLIGHT];
        let mut framebuffers = [FrameBuffer::default(); Self::FRAMES_IN_FLIGHT];
        for i in 0..Self::FRAMES_IN_FLIGHT {
            let allocation;
            let primary_color_buffer = unsafe {
                let create_info = vk::ImageCreateInfo {
                    queue_family_index_count: 1,
                    p_queue_family_indices: [vk.queue_family_index].as_ptr(),
                    flags: vk::ImageCreateFlags::empty(),
                    image_type: vk::ImageType::TYPE_2D,
                    format: hdr_color_format,
                    extent,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                };

                let image = vk.device.create_image(&create_info, vkdevice::MEMORY_ALLOCATOR).unwrap();
                allocation = vkdevice::allocate_image_memory(vk, image);
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
                    components: vkdevice::COMPONENT_MAPPING_DEFAULT,
                    subresource_range: image_subresource_range,
                    ..Default::default()
                };

                vk.device.create_image_view(&view_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
            };
            color_buffer_views[i] = color_buffer_view;
        
            //Create framebuffer
            let framebuffer_object = unsafe {
                let attachments = [color_buffer_view, depth_buffer_view];
                let fb_info = vk::FramebufferCreateInfo {
                    render_pass: hdr_render_pass,
                    attachment_count: attachments.len() as u32,
                    p_attachments: attachments.as_ptr(),
                    width: extent.width,
                    height: extent.height,
                    layers: 1,
                    ..Default::default()
                };
                vk.device.create_framebuffer(&fb_info, vkdevice::MEMORY_ALLOCATOR).unwrap()
            };
            hdr_framebuffers[i] = framebuffer_object;

            let gpu_image = GPUImage {
                image: primary_color_buffer,
                view: Some(color_buffer_view),
                width: extent.width,
                height: extent.height,
                mip_count: 1,
                format: hdr_color_format,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sampler,
                allocation
            };
            let texture_index = global_images.insert(gpu_image) as u32;

            framebuffers[i] = FrameBuffer {
                framebuffer_object: hdr_framebuffers[i],
                color_buffer: color_buffers[i],
                color_buffer_view: color_buffer_views[i],
                depth_buffer: depth_buffer_image,
                depth_buffer_view,
                texture_index
            };
        }
        
        framebuffers
    }

    pub fn upload_gltf_model(&mut self, vk: &mut VulkanGraphicsDevice, data: &GLTFMeshData, pipeline: vk::Pipeline) -> ModelKey {
        fn load_prim_png(vk: &mut VulkanGraphicsDevice, renderer: &mut Renderer, data: &GLTFMeshData, tex_id_map: &mut HashMap<usize, u32>, prim_tex_idx: usize) -> u32 {
            match tex_id_map.get(&prim_tex_idx) {
                Some(id) => { *id }
                None => {
                    let image = GPUImage::from_png_bytes(vk, renderer.material_sampler, data.texture_bytes[prim_tex_idx].as_slice());
                    let global_tex_id = renderer.global_images.insert(image) as u32;
                    tex_id_map.insert(prim_tex_idx, global_tex_id);
                    global_tex_id
                }
            }
        }

        //Compute this model's id
        let mut hasher = DefaultHasher::new();
        data.name.hash(&mut hasher);
        let id = hasher.finish();

        //Check if this model is already loaded
        if let Some(key) = self.models.id_exists(id) {
            self.models.increment_model_count(key);
            return key;
        }

        //let mut loading_images = vec![];
        let mut primitive_keys = vec![];
        let mut tex_id_map = HashMap::new();
        for prim in &data.primitives {
            let prim_tex_indices = [
                prim.material.color_index,
                prim.material.normal_index,
                prim.material.metallic_roughness_index,
                prim.material.emissive_index
            ];
            let mut inds = [None; 4];
            for i in 0..prim_tex_indices.len() {
                if let Some(idx) = prim_tex_indices[i] {
                    inds[i] = Some(load_prim_png(vk, self, data, &mut tex_id_map, idx));
                }
            }

            let material = Material {
                pipeline,
                base_color: prim.material.base_color,
                base_roughness: prim.material.base_roughness,
                base_metalness: prim.material.base_metalness,
                color_idx: inds[0],
                normal_idx: inds[1],
                metal_roughness_idx: inds[2],
                emissive_idx: inds[3]
            };
            let material_idx = self.global_materials.insert(material) as u32;

            let offsets = upload_primitive_vertices(vk, self, prim);

            let index_buffer = make_index_buffer(vk, &prim.indices);
            let model_idx = self.register_primitive(Primitive {
                shadow_type: ShadowType::OpaqueCaster,
                index_buffer,
                index_count: prim.indices.len().try_into().unwrap(),
                position_offset: offsets.position_offset,
                tangent_offset: offsets.tangent_offset,
                normal_offset: offsets.normal_offset,
                uv_offset: offsets.uv_offset,
                material_idx
            });
            primitive_keys.push(model_idx);
        }
        let model_key = self.new_model(id, primitive_keys);
        
        model_key
    }

    pub fn upload_ozymesh(&mut self, vk: &mut VulkanGraphicsDevice, data: &OzyMesh, pipeline: vk::Pipeline) -> ModelKey {
        fn load_prim_bc7(vk: &mut VulkanGraphicsDevice, renderer: &mut Renderer, data: &OzyMesh, tex_id_map: &mut HashMap<usize, u32>, prim_tex_idx: usize, format: vk::Format) -> u32 {
            match tex_id_map.get(&prim_tex_idx) {
                Some(id) => { *id }
                None => {
                    let width = data.textures[prim_tex_idx].width;
                    let height = data.textures[prim_tex_idx].height;
                    let mipmap_count = data.textures[prim_tex_idx].mipmap_count;
                    let raw_bytes = &data.textures[prim_tex_idx].bc7_bytes;
                    let image = GPUImage::from_bc7_bytes(vk, raw_bytes, renderer.material_sampler, width, height, mipmap_count, format);
                    let global_tex_id = renderer.global_images.insert(image) as u32;
                    tex_id_map.insert(prim_tex_idx, global_tex_id);
                    global_tex_id
                }
            }
        }

        //Compute this model's id
        let mut hasher = DefaultHasher::new();
        data.name.hash(&mut hasher);
        let id = hasher.finish();

        //Check if this model is already loaded
        if let Some(key) = self.models.id_exists(id) {
            self.models.increment_model_count(key);
            return key;
        }
        
        let mut primitive_keys = vec![];
        let mut tex_id_map = HashMap::new();
        for prim in &data.primitives {
            let ozy_material = &data.materials[prim.material_idx as usize];
            let prim_tex_indices = [
                ozy_material.color_bc7_idx,
                ozy_material.normal_bc7_idx,
                ozy_material.arm_bc7_idx,
                ozy_material.emissive_bc7_idx
            ];
            let mut inds = [None; 4];
            for i in 0..prim_tex_indices.len() {
                if let Some(idx) = prim_tex_indices[i] {
                    let format = vk::Format::BC7_UNORM_BLOCK;
                    inds[i] = Some(load_prim_bc7(vk, self, data, &mut tex_id_map, idx as usize, format));
                }
            }

            let material = Material {
                pipeline,
                base_color: ozy_material.base_color,
                base_roughness: ozy_material.base_roughness,
                base_metalness: ozy_material.base_metalness,
                color_idx: inds[0],
                normal_idx: inds[1],
                metal_roughness_idx: inds[2],
                emissive_idx: inds[3]
            };
            let material_idx = self.global_materials.insert(material) as u32;

            let offsets = upload_primitive_vertices(vk, self, prim);

            let index_buffer = make_index_buffer(vk, &prim.indices);
            let model_idx = self.register_primitive(Primitive {
                shadow_type: ShadowType::OpaqueCaster,
                index_buffer,
                index_count: prim.indices.len().try_into().unwrap(),
                position_offset: offsets.position_offset,
                tangent_offset: offsets.tangent_offset,
                normal_offset: offsets.normal_offset,
                uv_offset: offsets.uv_offset,
                material_idx
            });
            primitive_keys.push(model_idx);
        }
        let model_key = self.new_model(id, primitive_keys);
        
        model_key
    }

    pub fn new_model(&mut self, id: u64, primitive_keys: Vec<PrimitiveKey>) -> ModelKey {
        let model = Model {
            id,
            primitive_keys
        };
        self.models.insert(model)
    }

    pub fn delete_model(&mut self, key: ModelKey) {
        if let Some(model) = self.models.delete_instance(key) {
            for key in model.primitive_keys {
                self.delete_queue.push(DeferredDelete {
                    key,
                    frames_til_deletion: Self::FRAMES_IN_FLIGHT as u32
                });
            }
        }
    }

    pub fn new_directional_light(&mut self, light: SunLight) -> Option<DirectionalLightKey> {
        let mut ret = None;
        if self.directional_lights.len() < MAX_DIRECTIONAL_LIGHTS {
            ret = Some(self.directional_lights.insert(light));
        }
        ret
    }

    pub fn get_directional_light(&self, key: DirectionalLightKey) -> Option<&SunLight> {
        self.directional_lights.get(key)
    }

    pub fn get_directional_light_mut(&mut self, key: DirectionalLightKey) -> Option<&mut SunLight> {
        self.directional_lights.get_mut(key)
    }

    pub fn delete_directional_light(&mut self, key: DirectionalLightKey) {
        self.directional_lights.remove(key);
    }

    fn next_frame(&mut self, vk: &mut VulkanGraphicsDevice) -> InFlightFrameData {
        let cb = self.frames_in_flight[self.in_flight_frame];
        unsafe {
            vk.device.wait_for_fences(&[cb.fence], true, vk::DeviceSize::MAX).unwrap();
        }
        self.in_flight_frame += 1;
        self.in_flight_frame %= Self::FRAMES_IN_FLIGHT;
        cb
    }

    pub unsafe fn resize_hdr_framebuffers(&mut self, vk: &mut VulkanGraphicsDevice, extent: vk::Extent3D, hdr_render_pass: vk::RenderPass) {
        let fbs = self.framebuffers();
        for framebuffer in &fbs {
            vk.device.destroy_framebuffer(framebuffer.framebuffer_object, vkdevice::MEMORY_ALLOCATOR);
            vk.device.destroy_image_view(framebuffer.color_buffer_view, vkdevice::MEMORY_ALLOCATOR);
            vk.device.destroy_image(framebuffer.color_buffer, vkdevice::MEMORY_ALLOCATOR);
            self.global_images.remove(framebuffer.texture_index as usize);
        }
        vk.device.destroy_image_view(fbs[0].depth_buffer_view, vkdevice::MEMORY_ALLOCATOR);
        vk.device.destroy_image(fbs[0].depth_buffer, vkdevice::MEMORY_ALLOCATOR);

        let framebuffers = Self::create_hdr_framebuffers(vk, extent, hdr_render_pass, self.material_sampler, &mut self.global_images);
        for i in 0..self.frames_in_flight.len() {
            self.frames_in_flight[i].framebuffer = framebuffers[i];
        }
    }

    pub fn register_primitive(&mut self, data: Primitive) -> PrimitiveKey {
        self.primitives.insert(data)
    }

    fn upload_vertex_attribute(vk: &mut VulkanGraphicsDevice, data: &[f32], buffer: &GPUBuffer, offset: &mut u64) -> u32 {
        let old_offset = *offset;
        let new_offset = old_offset + data.len() as u64;
        buffer.write_subbuffer_elements(vk, data, old_offset);
        *offset = new_offset;
        old_offset.try_into().unwrap()
    }
    
    pub fn append_vertex_positions(&mut self, vk: &mut VulkanGraphicsDevice, positions: &[f32]) -> u32 {
        let buffer_block = BufferBlock {
            start_offset: self.position_offset,
            length: positions.len() as u64
        };
        let block_key = self.position_buffer_blocks.insert(buffer_block);
        Self::upload_vertex_attribute(vk, positions, &self.position_buffer, &mut self.position_offset) / 4
    }
    
    pub fn append_vertex_tangents(&mut self, vk: &mut VulkanGraphicsDevice, tangents: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, tangents, &self.tangent_buffer, &mut self.tangent_offset) / 4
    }
    
    pub fn append_vertex_normals(&mut self, vk: &mut VulkanGraphicsDevice, normals: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, normals, &self.normal_buffer, &mut self.normal_offset) / 4
    }
    
    pub fn append_vertex_uvs(&mut self, vk: &mut VulkanGraphicsDevice, uvs: &[f32]) -> u32 {
        Self::upload_vertex_attribute(vk, uvs, &self.uv_buffer, &mut self.uv_offset) / 2
    }

    pub fn replace_vertex_positions(&mut self, vk: &mut VulkanGraphicsDevice, data: &[f32], offset: u64) {
        let mut my_offset = offset * 4;
        Self::upload_vertex_attribute(vk, data, &self.position_buffer, &mut my_offset);
    }

    pub fn replace_vertex_tangents(&mut self, vk: &mut VulkanGraphicsDevice, data: &[f32], offset: u64) {
        let mut my_offset = offset * 4;
        Self::upload_vertex_attribute(vk, data, &self.tangent_buffer, &mut my_offset);
    }

    pub fn replace_vertex_normals(&mut self, vk: &mut VulkanGraphicsDevice, data: &[f32], offset: u64) {
        let mut my_offset = offset * 4;
        Self::upload_vertex_attribute(vk, data, &self.normal_buffer, &mut my_offset);
    }

    pub fn replace_vertex_uvs(&mut self, vk: &mut VulkanGraphicsDevice, data: &[f32], offset: u64) {
        let mut my_offset = offset * 2;
        Self::upload_vertex_attribute(vk, data, &self.uv_buffer, &mut my_offset);
    }

    pub fn replace_imgui_vertices(&mut self, vk: &mut VulkanGraphicsDevice, data: &[f32], offset: u64) {
        let mut my_offset = offset * DevGui::FLOATS_PER_VERTEX as u64;
        Self::upload_vertex_attribute(vk, data, &self.imgui_buffer, &mut my_offset);
    }

    pub fn get_primitive(&self, key: PrimitiveKey) -> Option<&Primitive> {
        self.primitives.get(key)
    }

    pub fn prepare_frame(&mut self, vk: &mut VulkanGraphicsDevice, window_size: glm::TVec2<u32>, view_from_world: &glm::TMat4<f32>, elapsed_time: f32) -> InFlightFrameData {
        //Process raw draw calls into draw stream
        {
            //Create map of ModelKeys to the instance data for all instances of that model
            let mut model_instances : HashMap<ModelKey, Vec<InstanceData>> = HashMap::new();
            for raw_draw in self.raw_draws.iter() {
                let mut instances = {
                    let mut is = Vec::with_capacity(raw_draw.world_transforms.len());
                    for i in 0..is.capacity() {
                        is.push(InstanceData::new(raw_draw.world_transforms[i]));
                    }
                    is
                };

                match model_instances.get_mut(&raw_draw.model_key) {
                    Some(data) => { data.append(&mut instances); }
                    None => { model_instances.insert(raw_draw.model_key, instances); }
                }
            }

            //Generate DrawCalls for each primitive of each instance
            let mut current_first_instance = 0;
            for (model_key, instances) in model_instances.iter_mut() {
                let instance_count = instances.len() as u32;
                if let Some(model) = self.models.get(*model_key) {
                    for prim_key in model.primitive_keys.iter() {
                        if let Some(primitive) = self.primitives.get(*prim_key) {
                            let pipeline = self.global_materials[primitive.material_idx.try_into().unwrap()].as_ref().unwrap().pipeline;
                            let dc = DrawCall {
                                primitive_key: *prim_key,
                                pipeline,
                                instance_count,
                                first_instance: current_first_instance
                            };
                            self.drawstream.push(dc);
                            self.instance_data.append(instances);
                        }
                    }
                }
                current_first_instance += instance_count;
            }

            //Sort DrawCalls according to their pipeline
            self.drawstream.sort_unstable();
        }

        //Process delete queue
        let mut i = 0;
        while i < self.delete_queue.len() {
            let item = &mut self.delete_queue[i];
            if item.frames_til_deletion == 0 {
                if let Some(primitive) = self.primitives.get(item.key) {
                    let mat_idx = primitive.material_idx.try_into().unwrap();
                    if let Some(material) = self.global_materials.remove(mat_idx) {
                        fn free_tex(vk: &mut VulkanGraphicsDevice, global_images: &mut FreeList<GPUImage>, index: Option<u32>) {
                            if let Some(idx) = index {
                                if let Some(image) = global_images.remove(idx.try_into().unwrap()) {
                                    image.free(vk);
                                }
                            }
                        }
                        free_tex(vk, &mut self.global_images, material.color_idx);
                        free_tex(vk, &mut self.global_images, material.normal_idx);
                        free_tex(vk, &mut self.global_images, material.metal_roughness_idx);
                        free_tex(vk, &mut self.global_images, material.emissive_idx);
                    }
                }

                self.delete_queue.remove(i);
            } else {
                item.frames_til_deletion -= 1;
                i += 1;
            }
        }

        //Wait for LRU frame to finish
        let frame_info = self.next_frame(vk);

        //Update bindless texture sampler descriptors
        if self.global_images.was_updated() {
            let default_texture = &self.global_images[self.default_texture_idx as usize].as_ref().unwrap();
            let default_descriptor_info = vk::DescriptorImageInfo {
                sampler: default_texture.sampler,
                image_view: default_texture.view.unwrap(),
                image_layout: default_texture.layout
            };

            let mut image_infos = vec![default_descriptor_info; self.global_images.size() as usize];
            for i in 0..self.global_images.len() {
                match &self.global_images[i] {
                    Some(image) => {
                        let descriptor_info = vk::DescriptorImageInfo {
                            sampler: image.sampler,
                            image_view: image.view.unwrap(),
                            image_layout: image.layout
                        };
                        image_infos[i] = descriptor_info;
                    }
                    None => {
                        image_infos[i] = default_descriptor_info;
                    }
                }
            }

            let sampler_write = vk::WriteDescriptorSet {
                dst_set: self.bindless_descriptor_set,
                descriptor_count: self.global_images.size() as u32,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: image_infos.as_ptr(),
                dst_array_element: 0,
                dst_binding: self.samplers_descriptor_index,
                ..Default::default()
            };
            unsafe { vk.device.update_descriptor_sets(&[sampler_write], &[]); }
        }

        //Update bindless material definitions
        if self.global_materials.was_updated() {
            let mut upload_mats = vec![MaterialData::default(); self.global_materials.len()];
            for i in 0..self.global_materials.len() {
                if let Some(mat) = &self.global_materials[i] {
                    upload_mats[i] = mat.data(self);
                }
            }

            self.material_buffer.write_buffer(vk, &upload_mats);
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

            let projection_matrix = glm::perspective_fov_rh_zo(
                glm::half_pi::<f32>(),
                window_size.x as f32,
                window_size.y as f32,
                Renderer::NEAR_CLIP_DISTANCE,
                Renderer::FAR_CLIP_DISTANCE
            );
            uniforms.clip_from_view = glm::mat4(
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ) * projection_matrix;
    
            uniforms.clip_from_world = uniforms.clip_from_view * view_from_world;

            //Push directional light data
            let mut i = 0;
            for light in self.directional_lights.iter() {
                let light = light.1;
                let irradiance = light.irradiance;
                let direction = 
                    glm::rotation(light.yaw, &glm::vec3(0.0, 0.0, 1.0)) *
                    glm::rotation(light.pitch, &glm::vec3(0.0, 1.0, 0.0)) *
                    glm::vec4(-1.0, 0.0, 0.0, 0.0);
                let direction = glm::vec4_to_vec3(&direction);

                uniforms.directional_lights[i] = DirectionalLight::new(direction, irradiance);

                let matrices = light.shadow_map.compute_shadow_cascade_matrices(
                    &direction,
                    &uniforms.view_from_world,
                    &uniforms.clip_from_view
                );
                for j in 0..matrices.len() {
                    uniforms.sun_shadow_matrices[CascadedShadowMap::CASCADE_COUNT * i + j] = matrices[j];
                }

                let dists = light.shadow_map.clip_distances();
                for j in 0..dists.len() {
                    uniforms.sun_shadow_distances[(CascadedShadowMap::CASCADE_COUNT + 1) * i + j] = dists[j];
                }

                i += 1;
            }
            uniforms.directional_light_count = self.directional_lights.len() as u32;
            
            //Compute the view-projection matrix for the skybox (the conversion functions are just there to nullify the translation component of the view matrix)
            //The skybox vertices should be rotated along with the camera, but they shouldn't be translated in order to maintain the illusion
            //that the sky is infinitely far away
            uniforms.clip_from_skybox = uniforms.clip_from_view * glm::mat3_to_mat4(&glm::mat4_to_mat3(&view_from_world));

            uniforms.time = elapsed_time;

            let dynamic_offset = (self.in_flight_frame as u64 * size_to_alignment!(size_of::<render::EnvironmentUniforms>() as u64, vk.physical_device_properties.limits.min_uniform_buffer_offset_alignment)) as u64;
            
            let uniform_bytes = struct_to_bytes(&self.uniform_data);
            self.uniform_buffer.write_subbuffer_elements(vk, uniform_bytes, dynamic_offset);
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
            self.instance_buffer.write_subbuffer_bytes(vk, instance_data_bytes, start_offset);
        }
        
        frame_info
    }

    pub fn drawcall(&mut self, model_key: ModelKey, world_transforms: Vec<glm::TMat4<f32>>) {
        let desired_draw = DesiredDraw {
            model_key,
            world_transforms
        };
        self.raw_draws.push(desired_draw);
    }

    pub fn drawlist_iter(&self) -> Iter<DrawCall> {
        self.drawstream.iter()
    }

    pub fn reset(&mut self) {
        self.raw_draws.clear();
        self.drawstream.clear();
        self.instance_data.clear();
    }
}
