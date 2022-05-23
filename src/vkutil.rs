use std::{ffi::c_void, io::Read, ops::Index};

use ash::vk;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;
use ozy::io::DDSHeader;
use sdl2::video::Window;
use std::ptr;
use crate::*;

pub const MEMORY_ALLOCATOR: Option<&vk::AllocationCallbacks> = None;

pub const COLOR_CLEAR: vk::ClearValue = {
    let color = vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 1.0]
        //float32: [0.26, 0.4, 0.46, 1.0]
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

pub const COMPONENT_MAPPING_DEFAULT: vk::ComponentMapping = vk::ComponentMapping {
    r: vk::ComponentSwizzle::R,
    g: vk::ComponentSwizzle::G,
    b: vk::ComponentSwizzle::B,
    a: vk::ComponentSwizzle::A,
};

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

pub unsafe fn allocate_image(vk: &mut VulkanAPI, image: vk::Image) -> Allocation {
    let requirements = vk.device.get_image_memory_requirements(image);
    let a = vk.allocator.allocate(&AllocationCreateDesc {
        name: &format!("VirtualImage {:?}", image),
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false       //We want tiled memory for images
    }).unwrap();

    vk.device.bind_image_memory(image, a.memory(), a.offset()).unwrap();
    a
}

//Macro to fit a given desired size to a given alignment without worrying about the specific integer type
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

pub enum ColorSpace {
    LINEAR,
    SRGB
}

pub struct VirtualImage {
    pub vk_image: vk::Image,
    pub vk_view: vk::ImageView,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub allocation: Allocation
}

impl VirtualImage {
    pub fn from_png_bytes(vk: &mut VulkanAPI, vk_command_buffer: vk::CommandBuffer, png_bytes: &[u8]) -> Self {
        use png::BitDepth;
        use png::ColorType;

        let decoder = png::Decoder::new(png_bytes);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();

        //We're given a depth in bits, so we set up an integer divide
        let byte_size_divisor = match info.bit_depth {
            BitDepth::One => { 8 }
            BitDepth::Two => { 4 }
            BitDepth::Four => { 2 }
            BitDepth::Eight => { 1 }
            _ => { crash_with_error_dialog("Unsupported PNG bitdepth"); }
        };

        match info.color_type {
            ColorType::Rgb => unsafe {
                let format = match info.srgb {
                    Some(_) => { vk::Format::R8G8B8A8_SRGB }
                    None => { vk::Format::R8G8B8A8_UNORM }
                };

                let width = info.width;
                let height = info.height;
                let pixel_count = (width * height / byte_size_divisor) as usize;
                let mut raw_bytes = vec![0u8; 3 * pixel_count];
                let mut bytes = vec![0xFFu8; 4 * pixel_count];
                reader.next_frame(&mut raw_bytes).unwrap();

                //Convert to RGBA by adding an alpha of 1.0 to each pixel
                for i in 0..pixel_count {
                    let idx = 4 * i;
                    let r_idx = 3 * i;
                    bytes[idx] = raw_bytes[r_idx];
                    bytes[idx + 1] = raw_bytes[r_idx + 1];
                    bytes[idx + 2] = raw_bytes[r_idx + 2];
                }
                
                let image_extent = vk::Extent3D {
                    width,
                    height,
                    depth: 1
                };
                let image_create_info = vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format,
                    extent: image_extent,
                    mip_levels: 1,
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
                let allocation = allocate_image(vk, image);

                let mut vim = VirtualImage {
                    vk_image: image,
                    vk_view: vk::ImageView::default(),
                    width,
                    height,
                    mip_count: 1,
                    allocation
                };
                vkutil::upload_image(vk, vk_command_buffer, &vim, &bytes);
                
                let sampler_subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                };
                let grass_view_info = vk::ImageViewCreateInfo {
                    image,
                    format,
                    view_type: vk::ImageViewType::TYPE_2D,
                    components: vkutil::COMPONENT_MAPPING_DEFAULT,
                    subresource_range: sampler_subresource_range,
                    ..Default::default()
                };
                let view = vk.device.create_image_view(&grass_view_info, vkutil::MEMORY_ALLOCATOR).unwrap();

                vim.vk_view = view;
                vim
            }
            t => { crash_with_error_dialog(&format!("Unsupported color type: {:?}", t)); }
        }
    }

    pub unsafe fn from_bc7(vk: &mut VulkanAPI, vk_command_buffer: vk::CommandBuffer, path: &str, color_space: ColorSpace) -> Self {
        let mut file = unwrap_result(File::open(path), &format!("Error opening image {}", path));
        let dds_header = DDSHeader::from_file(&mut file);       //This also advances the file read head to the beginning of the raw data section

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
        
        let format = match color_space {
            ColorSpace::LINEAR => {
                vk::Format::BC7_UNORM_BLOCK
            }
            ColorSpace::SRGB => {
                vk::Format::BC7_SRGB_BLOCK
            }
        };

        let image_extent = vk::Extent3D {
            width,
            height,
            depth: 1
        };
        let image_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
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
        let allocation = allocate_image(vk, image);

        let mut vim = VirtualImage {
            vk_image: image,
            vk_view: vk::ImageView::default(),
            width,
            height,
            mip_count: mipmap_count,
            allocation
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
            format,
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

pub unsafe fn upload_image(vk: &mut VulkanAPI, vk_command_buffer: vk::CommandBuffer, image: &VirtualImage, raw_bytes: &[u8]) {
    //Create staging buffer and upload raw image data
    let bytes_size = raw_bytes.len() as vk::DeviceSize;
    let staging_buffer = VirtualBuffer::allocate(vk, bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC);
    staging_buffer.upload_buffer(&raw_bytes);

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

    vk.device.cmd_copy_buffer_to_image(vk_command_buffer, staging_buffer.backing_buffer(), image.vk_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &copy_regions);

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
    vk.device.destroy_buffer(staging_buffer.backing_buffer(), vkutil::MEMORY_ALLOCATOR);
}

//All the variables that Vulkan needs
pub struct VulkanAPI {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub device: ash::Device,
    pub allocator: Allocator,
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
            use ash::vk::Handle;
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

                    let queue_create_info = vk::DeviceQueueCreateInfo {
                        queue_family_index: vk_queue_family_index,
                        queue_count: 1,
                        p_queue_priorities: [1.0].as_ptr(),
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

        //Initialize gpu_allocator
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: vk_instance.clone(),
            device: vk_device.clone(),
            physical_device: vk_physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,  // Ideally, check the BufferDeviceAddressFeatures struct.
        }).unwrap();

        VulkanAPI {
            instance: vk_instance,
            physical_device: vk_physical_device,
            physical_device_properties: vk_physical_device_properties,
            device: vk_device,
            allocator,
            surface: vk_surface,
            ext_surface: vk_ext_surface,
            queue_family_index: vk_queue_family_index
        }
    }
}

pub struct VirtualBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    length: vk::DeviceSize
}

impl VirtualBuffer {
    //Read-only access to fields
    pub fn allocation(&self) -> &Allocation { &self.allocation }
    pub fn backing_buffer(&self) -> vk::Buffer { self.buffer }
    pub fn length(&self) -> vk::DeviceSize { self.length }

    pub fn allocate(vk: &mut VulkanAPI, size: vk::DeviceSize, alignment: vk::DeviceSize, usage_flags: vk::BufferUsageFlags) -> Self {
        let vk_buffer;
        let actual_size = size_to_alignment!(size, alignment);
        let allocation = unsafe {
            let buffer_create_info = vk::BufferCreateInfo {
                usage: usage_flags,
                size: actual_size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            vk_buffer = vk.device.create_buffer(&buffer_create_info, vkutil::MEMORY_ALLOCATOR).unwrap();
            let mem_reqs = vk.device.get_buffer_memory_requirements(vk_buffer);

            let a = vk.allocator.allocate(&AllocationCreateDesc {
                name: "",
                requirements: mem_reqs,
                location: MemoryLocation::CpuToGpu,
                linear: true
            }).unwrap();
            vk.device.bind_buffer_memory(vk_buffer, a.memory(), a.offset()).unwrap();
            a
        };

        VirtualBuffer {
            buffer: vk_buffer,
            allocation,
            length: actual_size
        }
    }

    fn unchecked_ptr(&self) -> *mut c_void {
        self.allocation.mapped_ptr().unwrap().as_ptr()
    }

    pub fn upload_buffer<T>(&self, in_buffer: &[T]) {
        unsafe {
            let dst_ptr = self.unchecked_ptr();
            ptr::copy_nonoverlapping(in_buffer.as_ptr(), dst_ptr as *mut T, in_buffer.len());
        }
    }
}

pub struct VirtualGeometry {
    pub vertex_buffer: VirtualBuffer,
    pub index_buffer: VirtualBuffer,
    pub index_count: u32
}

impl VirtualGeometry {
    pub fn create(vk: &mut VulkanAPI, vertices: &[f32], indices: &[u32]) -> Self {
        let vertex_buffer = VirtualBuffer::allocate(vk, (vertices.len() * size_of::<f32>()) as vk::DeviceSize, 0, vk::BufferUsageFlags::VERTEX_BUFFER);
        let index_buffer = VirtualBuffer::allocate(vk, (indices.len() * size_of::<u32>()) as vk::DeviceSize, 0, vk::BufferUsageFlags::INDEX_BUFFER);
        vertex_buffer.upload_buffer(vertices);
        index_buffer.upload_buffer(indices);
        VirtualGeometry {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32
        }
    }
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
    pub fn initialize_swapchain(vk: &mut VulkanAPI, vk_ext_swapchain: &ash::extensions::khr::Swapchain, render_pass: vk::RenderPass) -> Self {
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

            let reqs = vk.device.get_image_memory_requirements(depth_image);
            let allocation = vk.allocator.allocate(&AllocationCreateDesc {
                name: "",
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: false       //We want tiled memory for images
            }).unwrap();

            //Bind the depth image to its memory
            vk.device.bind_image_memory(depth_image, allocation.memory(), allocation.offset()).unwrap();

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

pub struct VertexInputConfiguration<'a> {
    pub binding_descriptions: &'a [vk::VertexInputBindingDescription],
    pub attribute_descriptions: &'a [vk::VertexInputAttributeDescription]    
}

pub struct VirtualPipelineCreateInfo<'a> {
    render_pass: vk::RenderPass,
    vertex_config: VertexInputConfiguration<'a>,
    pub shader_stages: &'a [vk::PipelineShaderStageCreateInfo],
    pub rasterization_state: Option<vk::PipelineRasterizationStateCreateInfo>,
    pub depthstencil_state: Option<vk::PipelineDepthStencilStateCreateInfo>
}

impl<'a> VirtualPipelineCreateInfo<'a> {
    pub fn new(render_pass: vk::RenderPass, vertex_config: VertexInputConfiguration<'a>, shader_stages: &'a [vk::PipelineShaderStageCreateInfo]) -> Self {
        VirtualPipelineCreateInfo {
            render_pass,
            vertex_config,
            shader_stages,
            rasterization_state: None,
            depthstencil_state: None
        }
    }
}

pub struct PipelineCreator {
    pub default_dynamic_state_enables: [vk::DynamicState; 2],
    pub default_input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo,
    pub default_rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    pub default_color_blend_attachment_state: vk::PipelineColorBlendAttachmentState,
    pub default_viewport_state: vk::PipelineViewportStateCreateInfo,
    pub default_depthstencil_state: vk::PipelineDepthStencilStateCreateInfo,
    pub default_multisample_state: vk::PipelineMultisampleStateCreateInfo,
    pub pipeline_layout: vk::PipelineLayout
}

impl PipelineCreator {
    pub fn init(pipeline_layout: vk::PipelineLayout) -> Self {
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

        PipelineCreator {
            default_dynamic_state_enables: dynamic_state_enables,
            default_input_assembly_state: input_assembly_state,
            default_rasterization_state: rasterization_state,
            default_color_blend_attachment_state: color_blend_attachment_state,
            default_depthstencil_state: depth_stencil_state,
            default_multisample_state: multisample_state,
            default_viewport_state: viewport_state,
            pipeline_layout
        }
    }

    pub unsafe fn create_pipeline(&self, vk: &VulkanAPI, create_info: &VirtualPipelineCreateInfo) -> vk::Pipeline {

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: create_info.vertex_config.binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: create_info.vertex_config.binding_descriptions.as_ptr(),
            vertex_attribute_description_count: create_info.vertex_config.attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: create_info.vertex_config.attribute_descriptions.as_ptr(),
            ..Default::default()
        };

        //Handle any overrides in create_info
        let rasterization_state = match create_info.rasterization_state {
            Some(s) => { s }
            None => { self.default_rasterization_state }
        };
        let depthstencil_state = match create_info.depthstencil_state {
            Some(s) => { s }
            None => { self.default_depthstencil_state }
        };

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            p_dynamic_states: self.default_dynamic_state_enables.as_ptr(),
            dynamic_state_count: self.default_dynamic_state_enables.len() as u32,
            ..Default::default()
        };

        let color_blend_pipeline_state = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &self.default_color_blend_attachment_state,
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::NO_OP,
            blend_constants: [0.0; 4],
            ..Default::default()
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            layout: self.pipeline_layout,
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &self.default_input_assembly_state,
            p_rasterization_state: &rasterization_state,
            p_color_blend_state: &color_blend_pipeline_state,
            p_multisample_state: &self.default_multisample_state,
            p_dynamic_state: &dynamic_state,
            p_viewport_state: &self.default_viewport_state,
            p_depth_stencil_state: &depthstencil_state,
            p_stages: create_info.shader_stages.as_ptr(),
            stage_count: create_info.shader_stages.len() as u32,
            render_pass: create_info.render_pass,
            ..Default::default()
        };
        
        let pipeline = vk.device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], vkutil::MEMORY_ALLOCATOR).unwrap()[0];
        pipeline
    }

}

#[derive(Debug)]
pub struct FreeList<T> {
    list: OptionVec<T>,
    size: u64,
    pub updated: bool
}

impl<T> FreeList<T> {
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
}

impl<T> Index<usize> for FreeList<T> {
    type Output = Option<T>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.list[idx]
    }
}
