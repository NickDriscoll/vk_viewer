use std::{ffi::c_void, io::Read};

use ash::vk;
use ozy::io::DDSHeader;
use sdl2::video::Window;
use std::ptr;
use crate::*;

pub const MEMORY_ALLOCATOR: Option<&vk::AllocationCallbacks> = None;

pub const COLOR_CLEAR: vk::ClearValue = {
    let color = vk::ClearColorValue {
        //float32: [0.0, 0.0, 0.0, 1.0]
        float32: [0.26, 0.4, 0.46, 1.0]
    };
    vk::ClearValue {
        color
    }
};

pub const DEPTH_STENCIL_CLEAR: vk::ClearValue = {
    let value = vk::ClearDepthStencilValue {
        depth: 1.0,
        stencil: 0
    };
    vk::ClearValue {
        depth_stencil: value
    }
};

unsafe fn get_memory_type_index(
    vk_instance: &ash::Instance,
    vk_physical_device: vk::PhysicalDevice,
    memory_requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags
) -> Option<u32> {
    let mut i = 0;
    let mut memory_type_index = None;
    let mut largest_heap = 0;
    let phys_device_mem_props = vk_instance.get_physical_device_memory_properties(vk_physical_device);
    for mem_type in phys_device_mem_props.memory_types {
        if memory_requirements.memory_type_bits & (1 << i) != 0 && mem_type.property_flags.contains(flags) {
            let heap_size = phys_device_mem_props.memory_heaps[mem_type.heap_index as usize].size;
            if heap_size > largest_heap {
                memory_type_index = Some(i);
                largest_heap = heap_size;
            }
        }
        i += 1;
    }

    memory_type_index
}

pub unsafe fn allocate_buffer_memory(vk: &VulkanAPI, buffer: vk::Buffer) -> vk::DeviceMemory {
    let mem_reqs = vk.device.get_buffer_memory_requirements(buffer);
    let memory_type_index = get_memory_type_index(
        &vk.instance,
        vk.physical_device,
        mem_reqs,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    );
    if let None = memory_type_index {
        crash_with_error_dialog("Staging buffer memory allocation failed.");
    }
    let memory_type_index = memory_type_index.unwrap();

    let alloc_info = vk::MemoryAllocateInfo {
        allocation_size: mem_reqs.size,
        memory_type_index,
        ..Default::default()
    };
    vk.device.allocate_memory(&alloc_info, vkutil::MEMORY_ALLOCATOR).unwrap()    
}

pub unsafe fn allocate_image_memory(vk: &VulkanAPI, image: vk::Image) -> vk::DeviceMemory {
    let mem_reqs = vk.device.get_image_memory_requirements(image);

    //Search for the largest DEVICE_LOCAL heap the device advertises
    let memory_type_index = get_memory_type_index(&vk.instance, vk.physical_device, mem_reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL);
    if let None = memory_type_index {
        crash_with_error_dialog("Image memory allocation failed.");
    }
    let memory_type_index = memory_type_index.unwrap();

    let allocate_info = vk::MemoryAllocateInfo {
        allocation_size: mem_reqs.size,
        memory_type_index,
        ..Default::default()
    };
    vk.device.allocate_memory(&allocate_info, vkutil::MEMORY_ALLOCATOR).unwrap()
}

pub unsafe fn load_shader_stage(vk_device: &ash::Device, shader_stage_flags: vk::ShaderStageFlags, path: &str) -> vk::PipelineShaderStageCreateInfo {
    let mut file = File::open(path).unwrap();
    let spv = ash::util::read_spv(&mut file).unwrap();

    let module_create_info = vk::ShaderModuleCreateInfo {
        code_size: spv.len() * size_of::<u32>(),
        p_code: spv.as_ptr(),
        ..Default::default()
    };
    let module = vk_device.create_shader_module(&module_create_info, vkutil::MEMORY_ALLOCATOR).unwrap();

    vk::PipelineShaderStageCreateInfo {
        stage: shader_stage_flags,
        p_name: "main\0".as_ptr() as *const i8,
        module,
        ..Default::default()
    }
}

macro_rules! size_to_alignment {
    ($in_size:ident, $alignment:expr) => {
        {
            let mut final_size = $in_size;
            if $alignment > 0 {
                final_size = (final_size + ($alignment - 1)) & !($alignment - 1);   //Alignment is 2^N where N is a whole number
            }
            final_size
        }
    };
}

pub struct VirtualImage {
    pub vk_image: vk::Image,
    pub vk_view: vk::ImageView,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32
}

impl VirtualImage {
    pub unsafe fn from_bc7(vk: &VulkanAPI, vk_command_buffer: vk::CommandBuffer, path: &str) -> Self {
        let mut file = unwrap_result(File::open(path));
        let dds_header = DDSHeader::from_file(&mut file);

        let width = dds_header.width;
        let height = dds_header.height;
        let mipmap_count = dds_header.mipmap_count;

        let mut bytes_size = 0;
        for i in 0..mipmap_count {
            let w = width / (1 << i);
            let h = height / (1 << i);
            bytes_size += w * h;

            bytes_size = size_to_alignment!(bytes_size, 16);
        }

        let mut raw_bytes = vec![0u8; bytes_size as usize];
        file.read_exact(&mut raw_bytes).unwrap();
        
        let image_extent = vk::Extent3D {
            width,
            height,
            depth: 1
        };
        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::BC7_SRGB_BLOCK,
            extent: image_extent,
            mip_levels: mipmap_count,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &vk.queue_family_index,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let image = vk.device.create_image(&image_create_info, vkutil::MEMORY_ALLOCATOR).unwrap();

        let mut vim = VirtualImage {
            vk_image: image,
            vk_view: vk::ImageView::default(),
            width,
            height,
            mip_count: mipmap_count
        };
        upload_image(vk, vk_command_buffer, &vim, &raw_bytes);
        
        let sampler_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mipmap_count,
            base_array_layer: 0,
            layer_count: 1
        };
        let grass_view_info = vk::ImageViewCreateInfo {
            image: image,
            format: vk::Format::BC7_SRGB_BLOCK,
            view_type: vk::ImageViewType::TYPE_2D,
            components: COMPONENT_MAPPING_DEFAULT,
            subresource_range: sampler_subresource_range,
            ..Default::default()
        };
        let view = vk.device.create_image_view(&grass_view_info, vkutil::MEMORY_ALLOCATOR).unwrap();

        vim.vk_view = view;
        vim
    }
}

pub unsafe fn upload_image(vk: &VulkanAPI, vk_command_buffer: vk::CommandBuffer, image: &VirtualImage, raw_bytes: &[u8]) {
    let bytes_size = raw_bytes.len();

    let buffer_create_info = vk::BufferCreateInfo {
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        size: bytes_size as vk::DeviceSize,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let staging_buffer = vk.device.create_buffer(&buffer_create_info, vkutil::MEMORY_ALLOCATOR).unwrap();            
    let staging_buffer_memory = vkutil::allocate_buffer_memory(&vk, staging_buffer);    
    vk.device.bind_buffer_memory(staging_buffer, staging_buffer_memory, 0).unwrap();

    let staging_ptr = vk.device.map_memory(staging_buffer_memory, 0, bytes_size as vk::DeviceSize, vk::MemoryMapFlags::empty()).unwrap();
    ptr::copy_nonoverlapping(raw_bytes.as_ptr(), staging_ptr as *mut _, bytes_size as usize);
    vk.device.unmap_memory(staging_buffer_memory);

    let image_memory = allocate_image_memory(vk, image.vk_image);

    vk.device.bind_image_memory(image.vk_image, image_memory, 0).unwrap();

    vk.device.begin_command_buffer(vk_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: image.mip_count,
        base_array_layer: 0,
        layer_count: 1
    };
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image: image.vk_image,
        subresource_range,
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(vk_command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    let mut cumulative_offset = 0;
    let mut copy_regions = vec![vk::BufferImageCopy::default(); image.mip_count as usize];
    for i in 0..image.mip_count {
        let w = image.width / (1 << i);
        let h = image.height / (1 << i);
        let subresource_layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: i,
            base_array_layer: 0,
            layer_count: 1

        };
        let image_extent = vk::Extent3D {
            width: w,
            height: h,
            depth: 1
        };

        cumulative_offset = size_to_alignment!(cumulative_offset, 16);
        let copy_region = vk::BufferImageCopy {
            buffer_offset: cumulative_offset as u64,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_extent,
            image_offset: vk::Offset3D::default(),
            image_subresource: subresource_layers
        };
        copy_regions[i as usize] = copy_region;
        cumulative_offset += w * h;
    }

    vk.device.cmd_copy_buffer_to_image(vk_command_buffer, staging_buffer, image.vk_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copy_regions);

    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: image.mip_count,
        base_array_layer: 0,
        layer_count: 1
    };
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        image: image.vk_image,
        subresource_range,
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(vk_command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    vk.device.end_command_buffer(vk_command_buffer).unwrap();
    
    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &vk_command_buffer,
        ..Default::default()
    };

    let fence = vk.device.create_fence(&vk::FenceCreateInfo::default(), vkutil::MEMORY_ALLOCATOR).unwrap();
    let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
    vk.device.queue_submit(queue, &[submit_info], fence).unwrap();
    vk.device.wait_for_fences(&[fence], true, vk::DeviceSize::MAX).unwrap();
    vk.device.destroy_fence(fence, vkutil::MEMORY_ALLOCATOR);
    vk.device.destroy_buffer(staging_buffer, vkutil::MEMORY_ALLOCATOR);
}

//All the variables that Vulkan needs
pub struct VulkanAPI {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub device: ash::Device,
    pub surface: vk::SurfaceKHR,
    pub ext_surface: ash::extensions::khr::Surface,
    pub queue_family_index: u32
}

impl VulkanAPI {
    pub fn initialize(window: &Window) -> Self {
        let vk_entry = ash::Entry::linked();
        let vk_instance = {
            let app_info = vk::ApplicationInfo {
                api_version: vk::make_api_version(0, 1, 2, 0),
                ..Default::default()
            };

            #[cfg(target_os = "windows")]
            let extension_names = [
                ash::extensions::khr::Surface::name().as_ptr(),
                ash::extensions::khr::Win32Surface::name().as_ptr(),
            ];

            #[cfg(target_os = "macos")]
            let extension_names = [
                ash::extensions::khr::Surface::name().as_ptr(),
                ash::extensions::mvk::MacOSSurface::name().as_ptr()
            ];

            #[cfg(target_os = "linux")]
            let extension_names = [
                ash::extensions::khr::Surface::name().as_ptr(),
                ash::extensions::khr::XlibSurface::name().as_ptr()
            ];

            let layer_names = unsafe  {[
                CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()
            ]};
            let vk_create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_extension_count: extension_names.len() as u32,
                pp_enabled_extension_names: &extension_names as *const *const _,
                enabled_layer_count: layer_names.len() as u32,
                pp_enabled_layer_names: &layer_names as *const *const _,
                ..Default::default()
            };

            unsafe { vk_entry.create_instance(&vk_create_info, vkutil::MEMORY_ALLOCATOR).unwrap() }
        };

        //Use SDL to create the Vulkan surface
        let vk_surface = {
            let raw_surf = window.vulkan_create_surface(vk_instance.handle().as_raw() as usize).unwrap();
            vk::SurfaceKHR::from_raw(raw_surf)
        };
        
        let vk_ext_surface = ash::extensions::khr::Surface::new(&vk_entry, &vk_instance);

        //Create the Vulkan device
        let vk_physical_device;
        let vk_physical_device_properties;
        let mut vk_queue_family_index = 0;
        let vk_device = unsafe {
            match vk_instance.enumerate_physical_devices() {
                Ok(phys_devices) => {
                    let mut phys_device = None;
                    let device_types = [vk::PhysicalDeviceType::DISCRETE_GPU, vk::PhysicalDeviceType::INTEGRATED_GPU, vk::PhysicalDeviceType::CPU];
                    'gpu_search: for d_type in device_types {
                        for device in phys_devices.iter() {
                            let props = vk_instance.get_physical_device_properties(*device);
                            if props.device_type == d_type {
                                let name = CStr::from_ptr(props.device_name.as_ptr()).to_str().unwrap();
                                println!("\"{}\" was chosen as 3D accelerator.", name);
                                phys_device = Some(*device);
                                break 'gpu_search;
                            }
                        }
                    }

                    vk_physical_device = phys_device.unwrap();
                    vk_physical_device_properties = vk_instance.get_physical_device_properties(vk_physical_device);
                    
                    let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
                    let mut physical_device_features = vk::PhysicalDeviceFeatures2 {
                        p_next: &mut indexing_features as *mut _ as *mut c_void,
                        ..Default::default()
                    };
                    vk_instance.get_physical_device_features2(vk_physical_device, &mut physical_device_features);
                    if physical_device_features.features.texture_compression_bc == vk::FALSE {
                        println!("WARNING: GPU compressed textures not supported by this GPU.");
                    }

                    let mut i = 0;
                    for qfp in vk_instance.get_physical_device_queue_family_properties(vk_physical_device) {
                        if qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                            vk_queue_family_index = i;
                            break;
                        }
                        i += 1;
                    }

                    let priorities = [1.0];
                    let queue_create_info = vk::DeviceQueueCreateInfo {
                        queue_family_index: vk_queue_family_index,
                        queue_count: 1,
                        p_queue_priorities: priorities.as_ptr(),
                        ..Default::default()
                    };
                    
                    if !vk_ext_surface.get_physical_device_surface_support(vk_physical_device, vk_queue_family_index, vk_surface).unwrap() {
                        panic!("The physical device queue doesn't support swapchain present!");
                    }

                    let extension_names = [ash::extensions::khr::Swapchain::name().as_ptr()];
                    let create_info = vk::DeviceCreateInfo {
                        queue_create_info_count: 1,
                        p_queue_create_infos: [queue_create_info].as_ptr(),
                        enabled_extension_count: extension_names.len() as u32,
                        pp_enabled_extension_names: extension_names.as_ptr(),
                        p_enabled_features: &physical_device_features.features,
                        p_next: &mut indexing_features as *mut _ as *mut c_void,
                        ..Default::default()
                    };

                    vk_instance.create_device(vk_physical_device, &create_info, vkutil::MEMORY_ALLOCATOR).unwrap()
                }
                Err(e) => {
                    crash_with_error_dialog(&format!("Unable to enumerate physical devices: {}", e));
                }
            }
        };

        VulkanAPI {
            instance: vk_instance,
            physical_device: vk_physical_device,
            physical_device_properties: vk_physical_device_properties,
            device: vk_device,
            surface: vk_surface,
            ext_surface: vk_ext_surface,
            queue_family_index: vk_queue_family_index

        }
    }
}

#[derive(Clone, Copy)]
pub struct VirtualBuffer {
    backing_buffer: vk::Buffer,
    buffer_ptr: *mut c_void,
    offset: u64,
    length: u64
}

impl VirtualBuffer {
    //Read-only access to fields
    pub fn backing_buffer(&self) -> vk::Buffer { self.backing_buffer }
    pub fn offset(&self) -> u64 { self.offset }
    pub fn length(&self) -> u64 { self.length }

    pub fn upload_buffer<T>(&self, in_buffer: &[T]) {
        unsafe {
            let dst_ptr = self.buffer_ptr.offset(self.offset.try_into().unwrap());
            ptr::copy_nonoverlapping(in_buffer.as_ptr(), dst_ptr as *mut T, in_buffer.len());
        }
    }
}

impl Default for VirtualBuffer {
    fn default() -> VirtualBuffer {
        VirtualBuffer {
            backing_buffer: vk::Buffer::null(),
            buffer_ptr: ptr::null_mut(),
            offset: 0,
            length: 0
        }
    }
}

//Allocator that can only free its memory all at once
pub struct VirtualBumpAllocator {
    backing_buffer: vk::Buffer,
    buffer_ptr: *mut c_void,
    current_offset: u64,
    max_size: u64,
}

impl VirtualBumpAllocator {
    //Read-only access to fields
    //pub fn backing_buffer(&self) -> vk::Buffer { self.backing_buffer }
    //pub fn current_offset(&self) -> u64 { self.current_offset }
    //pub fn max_size(&self) -> u64 { self.max_size }

    pub fn new(backing_buffer: vk::Buffer, ptr: *mut c_void, max_size: u64) -> Self {
        VirtualBumpAllocator {
            backing_buffer,
            current_offset: 0,
            max_size,
            buffer_ptr: ptr
        }
    }

    pub fn clear(&mut self) {
        self.current_offset = 0;
    }


    pub fn allocate_buffer(&mut self, size: u64) -> Result<VirtualBuffer, String> {
        if size + self.current_offset > self.max_size {
            return Err(format!("Tried to allocate {} bytes from a buffer with {} bytes remaining", size, self.max_size - self.current_offset));
        }
        
        let b = VirtualBuffer {
            backing_buffer: self.backing_buffer,
            buffer_ptr: self.buffer_ptr,
            offset: self.current_offset,
            length: size
        };
        self.current_offset += size;
        Ok(b)
    }

    pub fn allocate_geometry(&mut self, v_buffer: &[f32], i_buffer: &[u32]) -> Result<VirtualGeometry, String> {
        let v_size = (v_buffer.len() * size_of::<f32>()) as u64;
        let i_size = (i_buffer.len() * size_of::<u32>()) as u64;
        let allocation_size = v_size + i_size;
        if allocation_size + self.current_offset > self.max_size {
            return Err(format!("Tried to allocate {} bytes from a buffer with {} bytes remaining", allocation_size, self.max_size - self.current_offset));
        }
        
        let vertex_buffer = self.allocate_buffer(v_size).unwrap();
        vertex_buffer.upload_buffer(&v_buffer);

        let index_buffer = self.allocate_buffer(i_size).unwrap();
        index_buffer.upload_buffer(&i_buffer);

        Ok (
            VirtualGeometry {
                vertex_buffer,
                index_buffer,
                index_count: i_buffer.len() as u32
            }
        )
    }
}

pub struct VirtualGeometry {
    pub vertex_buffer: VirtualBuffer,
    pub index_buffer: VirtualBuffer,
    pub index_count: u32
}

pub struct VirtualDrawCall<'a> {
    pub geometry: &'a VirtualGeometry,
    pub pipeline: vk::Pipeline,
    pub push_constants: [u8; 12],        //Assuming you get 12 fast bytes
    pub instance_count: u32,
    pub first_instance: u32
}

impl<'a> VirtualDrawCall<'a> {
    pub fn new(geometry: &'a VirtualGeometry, pipeline: vk::Pipeline, push_constants: [u32; 3], instance_count: u32, first_instance: u32) -> Self {
        let pcs = [push_constants[0].to_le_bytes(), push_constants[1].to_le_bytes(), push_constants[2].to_le_bytes()].concat();
        VirtualDrawCall {
            geometry,
            pipeline,
            push_constants: pcs.try_into().unwrap(),
            instance_count,
            first_instance
        }
    }
}

pub struct FrameUniforms {
    pub clip_from_screen: glm::TMat4<f32>,
    pub clip_from_world: glm::TMat4<f32>,
    pub clip_from_view: glm::TMat4<f32>,
    pub view_from_world: glm::TMat4<f32>,
    pub sun_direction: glm::TVec3<f32>,
    pub time: f32
}

pub struct Display {
    pub swapchain: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub color_format: vk::Format,
    pub depth_format: vk::Format,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_framebuffers: Vec<vk::Framebuffer>
}

impl Display {
    pub fn initialize_swapchain(vk: &VulkanAPI, vk_ext_swapchain: &ash::extensions::khr::Swapchain, render_pass: vk::RenderPass) -> Self {
        //Create the main swapchain for window present
        let vk_swapchain_image_format;
        let vk_swapchain_extent;
        let vk_swapchain = unsafe {
            let present_mode = vk.ext_surface.get_physical_device_surface_present_modes(vk.physical_device, vk.surface).unwrap()[0];
            let surf_capabilities = vk.ext_surface.get_physical_device_surface_capabilities(vk.physical_device, vk.surface).unwrap();
            let surf_formats = vk.ext_surface.get_physical_device_surface_formats(vk.physical_device, vk.surface).unwrap();

            //Search for an SRGB swapchain format
            let mut surf_format = vk::SurfaceFormatKHR::default();
            for sformat in surf_formats.iter() {
                if sformat.format == vk::Format::B8G8R8A8_SRGB {
                    surf_format = *sformat;
                    break;
                }
                surf_format = vk::SurfaceFormatKHR::default();
            }

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

            let sc = vk_ext_swapchain.create_swapchain(&create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            sc
        };
        
        let vk_swapchain_image_views = unsafe {
            let vk_swapchain_images = vk_ext_swapchain.get_swapchain_images(vk_swapchain).unwrap();

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
                    components: COMPONENT_MAPPING_DEFAULT,
                    subresource_range: image_subresource_range,
                    ..Default::default()
                };

                image_views.push(vk.device.create_image_view(&view_info, vkutil::MEMORY_ALLOCATOR).unwrap());
            }

            image_views
        };

        let vk_depth_format;
        let vk_depth_image = unsafe {
            let surf_capabilities = vk.ext_surface.get_physical_device_surface_capabilities(vk.physical_device, vk.surface).unwrap();
            let extent = vk::Extent3D {
                width: surf_capabilities.current_extent.width,
                height: surf_capabilities.current_extent.height,
                depth: 1
            };

            vk_depth_format = vk::Format::D24_UNORM_S8_UINT;
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
            let depth_memory = vkutil::allocate_image_memory(&vk, depth_image);

            //Bind the depth image to its memory
            vk.device.bind_image_memory(depth_image, depth_memory, 0).unwrap();

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
                components: COMPONENT_MAPPING_DEFAULT,
                subresource_range: image_subresource_range,
                ..Default::default()
            };

            vk.device.create_image_view(&view_info, vkutil::MEMORY_ALLOCATOR).unwrap()
        };

        //Create framebuffers
        let vk_swapchain_framebuffers = unsafe {
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

        Display {
            swapchain: vk_swapchain,
            extent: vk_swapchain_extent,
            color_format: vk_swapchain_image_format,
            depth_format: vk_depth_format,
            depth_image: vk_depth_image,
            swapchain_image_views: vk_swapchain_image_views,
            depth_image_view: vk_depth_image_view,
            swapchain_framebuffers: vk_swapchain_framebuffers
        }
    }
}