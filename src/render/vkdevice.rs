use std::{ffi::c_void, io::Read, ops::Index};

use ash::prelude::VkResult;
use ash::vk;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;
use ozy::io::DDSHeader;
use sdl2::libc::c_char;
use slotmap::new_key_type;
use std::ptr;
use crate::*;
use render::FreeList;

pub const MEMORY_ALLOCATOR: Option<&vk::AllocationCallbacks> = None;

pub const COLOR_CLEAR: vk::ClearValue = {
    let color = vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 1.0]           //float32: [0.26, 0.4, 0.46, 1.0]
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

//Macro to fit a given desired size to a given alignment without worrying about the specific integer type
#[macro_export]
macro_rules! size_to_alignment {
    ($in_size:expr, $alignment:expr) => {
        {
            let mut final_size = $in_size;
            if $alignment > 0 {
                final_size = (final_size + ($alignment - 1)) & !($alignment - 1);   //Alignment is 2^N where N is a whole number
            }
            final_size
        }
    };
}

pub struct VertexFetchOffsets {
    pub position_offset: u32,
    pub tangent_offset: u32,
    pub normal_offset: u32,
    pub uv_offset: u32,
}

pub fn msaa_samples_from_limit(sample_limit: vk::SampleCountFlags) -> vk::SampleCountFlags {
    if sample_limit.contains(vk::SampleCountFlags::TYPE_64) {
        vk::SampleCountFlags::TYPE_64
    } else if sample_limit.contains(vk::SampleCountFlags::TYPE_32) {
        vk::SampleCountFlags::TYPE_32
    } else if sample_limit.contains(vk::SampleCountFlags::TYPE_16) {
        vk::SampleCountFlags::TYPE_16
    } else if sample_limit.contains(vk::SampleCountFlags::TYPE_8) {
        vk::SampleCountFlags::TYPE_8
    } else if sample_limit.contains(vk::SampleCountFlags::TYPE_4) {
        vk::SampleCountFlags::TYPE_4
    } else if sample_limit.contains(vk::SampleCountFlags::TYPE_2) {
        vk::SampleCountFlags::TYPE_2
    } else {
        vk::SampleCountFlags::TYPE_1
    }
}

pub fn load_shader_stage(vk_device: &ash::Device, shader_stage_flags: vk::ShaderStageFlags, path: &str) -> vk::PipelineShaderStageCreateInfo {
    let msg = format!("Unable to read spv file {}\nDid a shader fail to compile?", path);
    let mut file = unwrap_result(File::open(path), &msg);
    let spv = unwrap_result(ash::util::read_spv(&mut file), &msg);

    let module_create_info = vk::ShaderModuleCreateInfo {
        code_size: spv.len() * size_of::<u32>(),
        p_code: spv.as_ptr(),
        ..Default::default()
    };
    let module = unsafe { vk_device.create_shader_module(&module_create_info, MEMORY_ALLOCATOR).unwrap() };

    vk::PipelineShaderStageCreateInfo {
        stage: shader_stage_flags,
        p_name: "main\0".as_ptr() as *const i8,
        module,
        ..Default::default()
    }
}

pub unsafe fn allocate_image_memory(vk: &mut VulkanGraphicsDevice, image: vk::Image) -> Allocation { allocate_named_image_memory(vk, image, "") }
pub unsafe fn allocate_named_image_memory(vk: &mut VulkanGraphicsDevice, image: vk::Image, name: &str) -> Allocation {
    let requirements = vk.device.get_image_memory_requirements(image);
    let alloc = vk.allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false       //We want tiled memory for images
    }).unwrap();

    vk.device.bind_image_memory(image, alloc.memory(), alloc.offset()).unwrap();
    alloc
}

pub fn load_bc7_texture(vk: &mut VulkanGraphicsDevice, global_textures: &mut FreeList<GPUImage>, sampler_key: SamplerKey, path: &str) -> u32 {
    unsafe {
        let vim = GPUImage::from_bc7_file(vk, sampler_key, path);
        global_textures.insert(vim) as u32
    }
}

pub unsafe fn upload_raw_image(vk: &mut VulkanGraphicsDevice, sampler_key: SamplerKey, format: vk::Format, layout: vk::ImageLayout, width: u32, height: u32, rgba: &[u8]) -> GPUImage {
    let image_create_info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 1,
        p_queue_family_indices: &vk.main_queue_family_index,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    let normal_image = vk.device.create_image(&image_create_info, MEMORY_ALLOCATOR).unwrap();
    let allocation = allocate_image_memory(vk, normal_image);

    let sampler = vk.get_sampler(sampler_key).unwrap();
    let mut vim = GPUImage {
        image: normal_image,
        view: None,
        width,
        height,
        mip_count: 1,
        format,
        layout,
        sampler,
        allocation
    };
    asset::upload_image(vk, &vim, &rgba);

    //Then create the image view
    let view_info = vk::ImageViewCreateInfo {
        image: normal_image,
        format,
        view_type: vk::ImageViewType::TYPE_2D,
        components: COMPONENT_MAPPING_DEFAULT,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1
        },
        ..Default::default()
    };
    vim.view = Some(vk.device.create_image_view(&view_info, MEMORY_ALLOCATOR).unwrap());
    
    vim
}

pub struct GPUImage {
    pub image: vk::Image,
    pub view: Option<vk::ImageView>,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub format: vk::Format,
    pub layout: vk::ImageLayout,
    pub sampler: vk::Sampler,
    pub allocation: Allocation
}

impl GPUImage {
    pub fn allocate(vk: &mut VulkanGraphicsDevice, create_info: &vk::ImageCreateInfo, sampler_key: SamplerKey) -> Self {
        unsafe {
            let image = vk.device.create_image(&create_info, MEMORY_ALLOCATOR).unwrap();
            let allocation = allocate_image_memory(vk, image);
            let width = create_info.extent.width;
            let height = create_info.extent.height;
            let mip_count = ozy::routines::calculate_mipcount(width, height);
            let sampler = vk.get_sampler(sampler_key).unwrap();

            GPUImage {
                image,
                view: None,
                width,
                height,
                mip_count,
                format: create_info.format,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sampler,
                allocation
            }
        }
    }

    pub fn free(self, vk: &mut VulkanGraphicsDevice) {
        vk.allocator.free(self.allocation).unwrap();
        unsafe {
            if let Some(view) = self.view {
                vk.device.destroy_image_view(view, MEMORY_ALLOCATOR);
            }
            vk.device.destroy_image(self.image, MEMORY_ALLOCATOR);
        }
    }

    pub fn from_png_file(vk: &mut VulkanGraphicsDevice, sampler_key: SamplerKey, path: &str) -> Self {
        let mut file = unwrap_result(File::open(path), &format!("Error opening png {}", path));
        let mut png_bytes = vec![0u8; file.metadata().unwrap().len().try_into().unwrap()];
        file.read_exact(&mut png_bytes).unwrap();
        Self::from_png_bytes(vk, sampler_key, &png_bytes)
    }

    pub fn from_png_file_deferred(vk: &mut VulkanGraphicsDevice, sampler_key: SamplerKey, path: &str) -> DeferredImage {
        let mut file = unwrap_result(File::open(path), &format!("Error opening png {}", path));
        let mut png_bytes = vec![0u8; file.metadata().unwrap().len().try_into().unwrap()];
        file.read_exact(&mut png_bytes).unwrap();
        Self::from_png_bytes_deferred(vk, sampler_key, &png_bytes)
    }
    pub fn from_png_bytes_deferred(vk: &mut VulkanGraphicsDevice, sampler_key: SamplerKey, png_bytes: &[u8]) -> DeferredImage {
        let decoder = png::Decoder::new(png_bytes);
        let read_info = decoder.read_info().unwrap();
        let info = read_info.info();
        let width = info.width;
        let height = info.height;
        let format = match info.srgb {
            Some(_) => { vk::Format::R8G8B8A8_SRGB }
            None => { vk::Format::R8G8B8A8_UNORM }
        };
        let bytes = asset::decode_png(read_info);

        unsafe {
            let mip_levels = ozy::routines::calculate_mipcount(width, height);
            let image_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1
                },
                mip_levels,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 1,
                p_queue_family_indices: &vk.main_queue_family_index,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            
            let mut def_image = asset::upload_image_deferred(vk, &image_create_info, sampler_key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, &bytes);

            let view_info = vk::ImageViewCreateInfo {
                image: def_image.final_image.image,
                format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: COMPONENT_MAPPING_DEFAULT,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1
                },
                ..Default::default()
            };
            def_image.final_image.view = Some(vk.device.create_image_view(&view_info, MEMORY_ALLOCATOR).unwrap());
            def_image
        }
    }

    pub fn from_png_bytes(vk: &mut VulkanGraphicsDevice, sampler_key: SamplerKey, png_bytes: &[u8]) -> Self {
        let decoder = png::Decoder::new(png_bytes);
        let read_info = decoder.read_info().unwrap();
        let info = read_info.info();
        let width = info.width;
        let height = info.height;
        let format = match info.srgb {
            Some(_) => { vk::Format::R8G8B8A8_SRGB }
            None => { vk::Format::R8G8B8A8_UNORM }
        };
        let bytes = asset::decode_png(read_info);
                
        //Create staging buffer and upload raw image data
        let bytes_size = bytes.len() as vk::DeviceSize;
        let staging_buffer = GPUBuffer::allocate(vk, bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
        staging_buffer.write_buffer(vk, &bytes);

        unsafe {
            let mip_levels = ozy::routines::calculate_mipcount(width, height);

            let image_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1
                },
                mip_levels,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 1,
                p_queue_family_indices: &vk.main_queue_family_index,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let image = vk.device.create_image(&image_create_info, MEMORY_ALLOCATOR).unwrap();
            let allocation = allocate_image_memory(vk, image);
            
            let view_info = vk::ImageViewCreateInfo {
                image,
                format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: COMPONENT_MAPPING_DEFAULT,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1
                },
                ..Default::default()
            };
            let view = vk.device.create_image_view(&view_info, MEMORY_ALLOCATOR).unwrap();

            let cbidx = vk.command_buffer_indices.insert(0);
            vk.device.begin_command_buffer(vk.command_buffers[cbidx], &vk::CommandBufferBeginInfo::default()).unwrap();

            let sampler = vk.get_sampler(sampler_key).unwrap();
            let mut vim = GPUImage {
                image,
                view: Some(view),
                width,
                height,
                mip_count: mip_levels,
                format,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sampler,
                allocation
            };
            asset::record_image_upload_commands(vk, vk.command_buffers[cbidx], &vim, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, &staging_buffer);

            vk.device.end_command_buffer(vk.command_buffers[cbidx]).unwrap();
    
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &vk.command_buffers[cbidx],
                ..Default::default()
            };
            let queue = vk.device.get_device_queue(vk.main_queue_family_index, 0);
            vk.device.wait_for_fences(&[vk.command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
            vk.device.reset_fences(&[vk.command_buffer_fence]).unwrap();
            vk.device.queue_submit(queue, &[submit_info], vk.command_buffer_fence).unwrap();
            vk.device.wait_for_fences(&[vk.command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
            vk.command_buffer_indices.remove(cbidx);

            vim
        }
    }

    pub fn from_bc7_bytes(vk: &mut VulkanGraphicsDevice, raw_bytes: &[u8], sampler_key: SamplerKey, width: u32, height: u32, mipmap_count: u32, format: vk::Format) -> Self {
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
            p_queue_family_indices: &vk.main_queue_family_index,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        unsafe {
            let image = vk.device.create_image(&image_create_info, MEMORY_ALLOCATOR).unwrap();
            let allocation = allocate_image_memory(vk, image);
            let sampler = vk.get_sampler(sampler_key).unwrap();

            let mut vim = GPUImage {
                image,
                view: None,
                width,
                height,
                mip_count: mipmap_count,
                format,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                sampler,
                allocation
            };
            asset::upload_image(vk, &vim, &raw_bytes);
            
            let view_info = vk::ImageViewCreateInfo {
                image: image,
                format,
                view_type: vk::ImageViewType::TYPE_2D,
                components: COMPONENT_MAPPING_DEFAULT,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mipmap_count,
                    base_array_layer: 0,
                    layer_count: 1
                },
                ..Default::default()
            };
            let view = vk.device.create_image_view(&view_info, MEMORY_ALLOCATOR).unwrap();

            vim.view = Some(view);
            vim
        }
    }

    pub unsafe fn from_bc7_file(vk: &mut VulkanGraphicsDevice, sampler_key: SamplerKey, path: &str) -> Self {
        use ozy::io::DXGI_FORMAT;

        let mut file = unwrap_result(File::open(path), &format!("Error opening bc7 {}", path));
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
        
        let format = match dds_header.dx10_header.dxgi_format {
            DXGI_FORMAT::BC7_UNORM => { vk::Format::BC7_UNORM_BLOCK }
            DXGI_FORMAT::BC7_UNORM_SRGB => { vk::Format::BC7_SRGB_BLOCK }
            _ => { crash_with_error_dialog("Unreachable statement reached in GPUImage::from_bc7_file()"); }
        };

        Self::from_bc7_bytes(vk, &raw_bytes, sampler_key, width, height, mipmap_count, format)
    }
}

pub unsafe fn upload_GPU_buffer<T>(vk: &mut VulkanGraphicsDevice, dst_buffer: vk::Buffer, offset: u64, raw_data: &[T]) {
    //Create staging buffer and upload raw buffer data
    let bytes_size = (raw_data.len() * size_of::<T>()) as vk::DeviceSize;
    let staging_buffer = GPUBuffer::allocate(vk, bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
    staging_buffer.write_buffer(vk, &raw_data);

    //Wait on the fence before beginning command recording
    vk.device.wait_for_fences(&[vk.command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    vk.device.reset_fences(&[vk.command_buffer_fence]).unwrap();
    let cbidx = vk.command_buffer_indices.insert(0);
    vk.device.begin_command_buffer(vk.command_buffers[cbidx], &vk::CommandBufferBeginInfo::default()).unwrap();

    let copy = vk::BufferCopy {
        src_offset: 0,
        dst_offset: offset * size_of::<T>() as u64,
        size: bytes_size
    };
    vk.device.cmd_copy_buffer(vk.command_buffers[cbidx], staging_buffer.backing_buffer(), dst_buffer, &[copy]);

    vk.device.end_command_buffer(vk.command_buffers[cbidx]).unwrap();

    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &vk.command_buffers[cbidx],
        ..Default::default()
    };
    let queue = vk.device.get_device_queue(vk.main_queue_family_index, 0);
    vk.device.queue_submit(queue, &[submit_info], vk.command_buffer_fence).unwrap();
    vk.device.wait_for_fences(&[vk.command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    vk.command_buffer_indices.remove(cbidx);
    staging_buffer.free(vk);
}

//All of the data in a DeferredReadback is valid when it's created, but the GPU-side
//data may only be considered valid after the associated fence has been signaled
//staging_buffer can only be freed after fence has been signaled
pub struct DeferredReadback {
    pub fence: vk::Fence,
    pub staging_buffer: GPUBuffer,
    pub command_buffer_idx: usize,
}

//All of the data in a DeferredImage is valid when it's created, but the GPU-side
//data may only be considered valid after the associated fence has been signaled
//staging_buffer can only be freed after fence has been signaled
pub struct DeferredImage {
    pub fence: vk::Fence,
    pub staging_buffer: Option<GPUBuffer>,
    pub command_buffer_idx: usize,
    pub final_image: GPUImage
}

impl DeferredImage {
    pub fn synchronize(vk: &mut VulkanGraphicsDevice, images: Vec<Self>) -> Vec<Self> {
        unsafe {
            let mut fences = Vec::with_capacity(images.len());
            for image in images.iter() {
                fences.push(image.fence);
            }
            vk.device.wait_for_fences(&fences, true, vk::DeviceSize::MAX).unwrap();
            let mut new_images = Vec::with_capacity(images.len());
            for mut image in images {
                if let Some(buffer) = image.staging_buffer {
                    buffer.free(vk);
                }
                image.staging_buffer = None;
                vk.command_buffer_indices.remove(image.command_buffer_idx);
                new_images.push(image);
            }
            new_images
        }
    }
}

//All the variables associated with the Vulkan graphics device
new_key_type! { pub struct SamplerKey; }
pub struct VulkanGraphicsDevice {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub device: ash::Device,
    pub allocator: Allocator,
    pub ext_surface: ash::extensions::khr::Surface,
    pub ext_swapchain: ash::extensions::khr::Swapchain,
    pub main_queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub command_buffer_indices: FreeList<u8>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub command_buffer_fence: vk::Fence,
    samplers: DenseSlotMap<SamplerKey, vk::Sampler>
}

impl VulkanGraphicsDevice {
    pub unsafe fn create_sampler(&mut self, info: &vk::SamplerCreateInfo) -> VkResult<SamplerKey> {
        let sampler = match self.device.create_sampler(info, MEMORY_ALLOCATOR) {
            Ok(s) => { s }
            Err(e) => { return Err(e) }
        };
        Ok(self.samplers.insert(sampler))
    }

    pub fn get_sampler(&self, key: SamplerKey) -> Option<vk::Sampler> {
        match self.samplers.get(key) {
            Some(k) => { Some(*k) }
            None => None
        }
    }

    pub unsafe fn destroy_sampler(&mut self, key: SamplerKey) -> bool {
        match self.samplers.remove(key) {
            Some(sampler) => {
                self.device.destroy_sampler(sampler, MEMORY_ALLOCATOR);
                true
            }
            None => { false }
        }
    }

    pub fn init() -> Self {
        let vk_entry = ash::Entry::linked();
        let vk_instance = {
            let app_info = vk::ApplicationInfo {
                api_version: vk::make_api_version(0, 1, 2, 0),
                ..Default::default()
            };

            #[cfg(target_os = "windows")]
            let platform_surface_extension = ash::extensions::khr::Win32Surface::name().as_ptr();
            
            #[cfg(target_os = "macos")]
            let platform_surface_extension = ash::extensions::mvk::MacOSSurface::name().as_ptr();

            #[cfg(target_os = "linux")]
            let platform_surface_extension = ash::extensions::khr::XlibSurface::name().as_ptr();

            let extension_names = [
                ash::extensions::khr::Surface::name().as_ptr(),
                platform_surface_extension
            ];
            
            let layer_names = [];
            
            let vk_create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_extension_count: extension_names.len() as u32,
                pp_enabled_extension_names: &extension_names as *const *const c_char,
                enabled_layer_count: layer_names.len() as u32,
                pp_enabled_layer_names: &layer_names as *const *const c_char,
                ..Default::default()
            };

            unsafe { vk_entry.create_instance(&vk_create_info, MEMORY_ALLOCATOR).expect("Crash during Vulkan instance creation") }
        };
        let vk_ext_surface = ash::extensions::khr::Surface::new(&vk_entry, &vk_instance);

        //Create the Vulkan device
        let vk_physical_device;
        let vk_physical_device_properties;
        let mut queue_family_index = 0;
        let buffer_device_address;
        let vk_device = unsafe {
            let phys_devices = vk_instance.enumerate_physical_devices();
            if let Err(e) = phys_devices {
                crash_with_error_dialog(&format!("Unable to enumerate physical devices: {}", e));
            }
            let phys_devices = phys_devices.unwrap();

            //Search for the physical device
            const DEVICE_TYPES: [vk::PhysicalDeviceType; 3] = [
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                vk::PhysicalDeviceType::CPU
            ];
            let mut phys_device = None;
            'gpu_search: for d_type in DEVICE_TYPES {
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

            vk_physical_device = match phys_device {
                Some(device) => { device }
                None => { crash_with_error_dialog("Unable to selected physical device."); }
            };

            vk_physical_device_properties = vk_instance.get_physical_device_properties(vk_physical_device);
            
            //Get physical device features
            let mut multiview_features = vk::PhysicalDeviceMultiviewFeatures::default();
            let mut buffer_address_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
            let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default();
            indexing_features.p_next = &mut buffer_address_features as *mut _ as *mut c_void;
            buffer_address_features.p_next = &mut multiview_features as *mut _ as *mut c_void;
            multiview_features.p_next = &mut sync2_features as *mut _ as *mut c_void;
            let mut physical_device_features = vk::PhysicalDeviceFeatures2 {
                p_next: &mut indexing_features as *mut _ as *mut c_void,
                ..Default::default()
            };
            vk_instance.get_physical_device_features2(vk_physical_device, &mut physical_device_features);
            
            if multiview_features.multiview == vk::FALSE {
                crash_with_error_dialog("Your GPU does not support multiview rendering.");
            }

            if physical_device_features.features.texture_compression_bc == vk::FALSE {
                tfd::message_box_ok("WARNING", "GPU compressed textures are not supported by this GPU.\nYou may be able to get away with this...", tfd::MessageBoxIcon::Warning);
            }
            buffer_device_address = buffer_address_features.buffer_device_address != vk::FALSE;
            
            if indexing_features.descriptor_binding_partially_bound == vk::FALSE || indexing_features.runtime_descriptor_array == vk::FALSE {
                crash_with_error_dialog("Your GPU lacks the specific features required to do bindless rendering. Sorry.");
            }

            let mut i = 0;
            let qfps = vk_instance.get_physical_device_queue_family_properties(vk_physical_device);
            for qfp in qfps {
                if qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    queue_family_index = i;
                    break;
                }
                i += 1;
            }

            let queue_create_info = vk::DeviceQueueCreateInfo {
                queue_family_index,
                queue_count: 1,
                p_queue_priorities: [1.0].as_ptr(),
                ..Default::default()
            };

            let extension_names = unsafe {[
                CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_non_semantic_info\0").as_ptr(),
                ash::extensions::khr::Swapchain::name().as_ptr()
            ]};
            let create_info = vk::DeviceCreateInfo {
                queue_create_info_count: 1,
                p_queue_create_infos: [queue_create_info].as_ptr(),
                enabled_extension_count: extension_names.len() as u32,
                pp_enabled_extension_names: extension_names.as_ptr(),
                p_enabled_features: &physical_device_features.features,
                p_next: &mut indexing_features as *mut _ as *mut c_void,
                ..Default::default()
            };

            vk_instance.create_device(vk_physical_device, &create_info, MEMORY_ALLOCATOR).expect("Crash during VkDevice creation")
        };
        
        let vk_ext_swapchain = ash::extensions::khr::Swapchain::new(&vk_instance, &vk_device);

        //Initialize gpu_allocator
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: vk_instance.clone(),
            device: vk_device.clone(),
            physical_device: vk_physical_device,
            debug_settings: Default::default(),
            buffer_device_address
        }).unwrap();

        let command_pool = unsafe {
            let pool_create_info = vk::CommandPoolCreateInfo {
                queue_family_index,
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            };
    
            vk_device.create_command_pool(&pool_create_info, MEMORY_ALLOCATOR).unwrap()
        };

        //Create command buffer
        let command_buffer_count = 1024;
        let general_command_buffers = unsafe {
            let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
                command_pool,
                command_buffer_count: command_buffer_count as u32,
                level: vk::CommandBufferLevel::PRIMARY,
                ..Default::default()
            };
            vk_device.allocate_command_buffers(&command_buffer_alloc_info).unwrap()
        };

        let graphics_command_buffer_fence = unsafe {
            let create_info = vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            };
            vk_device.create_fence(&create_info, MEMORY_ALLOCATOR).unwrap()
        };

        VulkanGraphicsDevice {
            instance: vk_instance,
            physical_device: vk_physical_device,
            physical_device_properties: vk_physical_device_properties,
            device: vk_device,
            allocator,
            ext_surface: vk_ext_surface,
            ext_swapchain: vk_ext_swapchain,
            main_queue_family_index: queue_family_index,
            command_pool,
            command_buffer_indices: FreeList::with_capacity(command_buffer_count),
            command_buffers: general_command_buffers,
            command_buffer_fence: graphics_command_buffer_fence,
            samplers: DenseSlotMap::with_key()
        }
    }
}

pub struct GPUBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    length: vk::DeviceSize
}

impl GPUBuffer {
    //Read-only access to fields
    pub fn backing_buffer(&self) -> vk::Buffer { self.buffer }
    pub fn length(&self) -> vk::DeviceSize { self.length }

    pub fn allocate(vk: &mut VulkanGraphicsDevice, size: vk::DeviceSize, alignment: vk::DeviceSize, usage_flags: vk::BufferUsageFlags, memory_location: MemoryLocation) -> Self {
        let vk_buffer;
        let actual_size = size_to_alignment!(size, alignment);
        let allocation = unsafe {
            let buffer_create_info = vk::BufferCreateInfo {
                usage: usage_flags,
                size: actual_size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            vk_buffer = vk.device.create_buffer(&buffer_create_info, MEMORY_ALLOCATOR).unwrap();
            let mem_reqs = vk.device.get_buffer_memory_requirements(vk_buffer);

            let a = vk.allocator.allocate(&AllocationCreateDesc {
                name: "",
                requirements: mem_reqs,
                location: memory_location,
                linear: true
            }).unwrap();
            vk.device.bind_buffer_memory(vk_buffer, a.memory(), a.offset()).unwrap();
            a
        };

        GPUBuffer {
            buffer: vk_buffer,
            allocation,
            length: actual_size
        }
    }

    pub fn free(self, vk: &mut VulkanGraphicsDevice) {
        vk.allocator.free(self.allocation).unwrap();
        unsafe { vk.device.destroy_buffer(self.buffer, MEMORY_ALLOCATOR); }
    }

    #[named]
    pub fn read_buffer_bytes(&self) -> Vec<u8> {
        match self.allocation.mapped_slice() {
            Some(s) => {
                s.to_vec()
            }
            None => {
                crash_with_error_dialog(&format!("{} is currently only implemented for host-mapped buffers", function_name!()));
            }
        }
    }

    pub fn write_buffer<T>(&self, vk: &mut VulkanGraphicsDevice, in_buffer: &[T]) {
        self.write_subbuffer_elements(vk, in_buffer, 0);
    }

    pub fn write_subbuffer_elements<T>(&self, vk: &mut VulkanGraphicsDevice, in_buffer: &[T], offset: u64) {
        let byte_buffer = slice_to_bytes(in_buffer);
        let byte_offset = offset * size_of::<T>() as u64;
        self.write_subbuffer_bytes(vk, byte_buffer, byte_offset as u64);
    }

    pub fn write_subbuffer_bytes(&self, vk: &mut VulkanGraphicsDevice, in_buffer: &[u8], offset: u64) {
        let end_in_bytes = in_buffer.len() + offset as usize;
        if end_in_bytes as u64 > self.length {
            crash_with_error_dialog("OVERRAN BUFFER AAAAA");
        }

        unsafe {
            match self.allocation.mapped_ptr() {
                Some(p) => {
                    let dst_ptr = p.as_ptr() as *mut u8;
                    let dst_ptr = dst_ptr.offset(offset as isize);
                    ptr::copy_nonoverlapping(in_buffer.as_ptr(), dst_ptr as *mut u8, in_buffer.len());
                }
                None => {
                    upload_GPU_buffer(vk, self.buffer, offset, in_buffer);
                }
            }
        }
    }
}