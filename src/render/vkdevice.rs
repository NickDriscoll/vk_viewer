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

pub struct VertexBlocks {
    pub position_block: GPUBufferBlock,
    pub tangent_block: GPUBufferBlock,
    pub normal_block: GPUBufferBlock,
    pub uv_block: GPUBufferBlock
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

pub unsafe fn allocate_image_memory(gpu: &mut VulkanGraphicsDevice, image: vk::Image) -> Allocation { allocate_named_image_memory(gpu, image, "") }
pub unsafe fn allocate_named_image_memory(gpu: &mut VulkanGraphicsDevice, image: vk::Image, name: &str) -> Allocation {
    let requirements = gpu.device.get_image_memory_requirements(image);
    let alloc = gpu.allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false,       //We want tiled memory for images
        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged
    }).unwrap();

    gpu.device.bind_image_memory(image, alloc.memory(), alloc.offset()).unwrap();
    alloc
}

pub unsafe fn upload_raw_image(gpu: &mut VulkanGraphicsDevice, sampler_key: SamplerKey, format: vk::Format, layout: vk::ImageLayout, width: u32, height: u32, rgba: &[u8]) -> GPUImage {
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
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 1,
        p_queue_family_indices: &gpu.main_queue_family_index,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    let normal_image = gpu.device.create_image(&image_create_info, MEMORY_ALLOCATOR).unwrap();
    let allocation = allocate_image_memory(gpu, normal_image);

    let sampler = gpu.get_sampler(sampler_key).unwrap();
    let mut vim = GPUImage {
        image: normal_image,
        view: None,
        width,
        height,
        mip_count: 1,
        format,
        layout,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        sampler,
        allocation
    };
    asset::upload_image(gpu, &vim, &rgba);

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
    vim.view = Some(gpu.device.create_image_view(&view_info, MEMORY_ALLOCATOR).unwrap());
    
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
    pub usage: vk::ImageUsageFlags,
    pub sampler: vk::Sampler,
    pub allocation: Allocation
}

impl GPUImage {
    pub fn allocate(gpu: &mut VulkanGraphicsDevice, create_info: &vk::ImageCreateInfo, sampler_key: SamplerKey) -> Self {
        unsafe {
            let image = gpu.device.create_image(&create_info, MEMORY_ALLOCATOR).unwrap();
            let allocation = allocate_image_memory(gpu, image);
            let width = create_info.extent.width;
            let height = create_info.extent.height;
            let usage = create_info.usage;
            let mip_count = create_info.mip_levels;
            let sampler = gpu.get_sampler(sampler_key).unwrap();

            GPUImage {
                image,
                view: None,
                width,
                height,
                mip_count,
                format: create_info.format,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                usage,
                sampler,
                allocation
            }
        }
    }

    pub fn free(self, gpu: &mut VulkanGraphicsDevice) {
        gpu.allocator.free(self.allocation).unwrap();
        unsafe {
            if let Some(view) = self.view {
                gpu.device.destroy_image_view(view, MEMORY_ALLOCATOR);
            }
            gpu.device.destroy_image(self.image, MEMORY_ALLOCATOR);
        }
    }
}

pub unsafe fn upload_GPU_buffer<T>(gpu: &mut VulkanGraphicsDevice, dst_buffer: vk::Buffer, offset: u64, raw_data: &[T]) {
    //Create staging buffer and upload raw buffer data
    let bytes_size = (raw_data.len() * size_of::<T>()) as vk::DeviceSize;
    let staging_buffer = GPUBuffer::allocate(gpu, bytes_size, 0, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
    staging_buffer.write_buffer(gpu, &raw_data);

    //Wait on the fence before beginning command recording
    gpu.device.wait_for_fences(&[gpu.command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    gpu.device.reset_fences(&[gpu.command_buffer_fence]).unwrap();
    let cbidx = gpu.command_buffer_indices.insert(0);
    gpu.device.begin_command_buffer(gpu.command_buffers[cbidx], &vk::CommandBufferBeginInfo::default()).unwrap();

    let copy = vk::BufferCopy {
        src_offset: 0,
        dst_offset: offset * size_of::<T>() as u64,
        size: bytes_size
    };
    gpu.device.cmd_copy_buffer(gpu.command_buffers[cbidx], staging_buffer.buffer(), dst_buffer, &[copy]);

    gpu.device.end_command_buffer(gpu.command_buffers[cbidx]).unwrap();

    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &gpu.command_buffers[cbidx],
        ..Default::default()
    };
    let queue = gpu.device.get_device_queue(gpu.main_queue_family_index, 0);
    gpu.device.queue_submit(queue, &[submit_info], gpu.command_buffer_fence).unwrap();
    gpu.device.wait_for_fences(&[gpu.command_buffer_fence], true, vk::DeviceSize::MAX).unwrap();
    gpu.command_buffer_indices.remove(cbidx);
    staging_buffer.free(gpu);
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
    pub gpu_image: GPUImage
}

impl DeferredImage {
    pub fn synchronize(gpu: &mut VulkanGraphicsDevice, images: Vec<Self>) -> Vec<Self> {
        unsafe {
            let mut fences = Vec::with_capacity(images.len());
            for image in images.iter() {
                fences.push(image.fence);
            }
            gpu.device.wait_for_fences(&fences, true, vk::DeviceSize::MAX).unwrap();
            let mut new_images = Vec::with_capacity(images.len());
            for mut image in images {
                if let Some(buffer) = image.staging_buffer {
                    buffer.free(gpu);
                }
                image.staging_buffer = None;
                gpu.command_buffer_indices.remove(image.command_buffer_idx);
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
    pub ext_sync2: ash::extensions::khr::Synchronization2,
    pub main_queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub command_buffer_indices: FreeList<u8>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub command_buffer_fence: vk::Fence,
    samplers: DenseSlotMap<SamplerKey, vk::Sampler>
}

impl VulkanGraphicsDevice {
    pub unsafe fn upload_image(&mut self, info: &vk::ImageCreateInfo, sampler_key: SamplerKey, bytes: &[u8]) -> DeferredImage {
        let mut def_image = asset::upload_image_deferred(self, &info, sampler_key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, &bytes);
        
        let view_type = match info.image_type {
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => { crash_with_error_dialog("Unreachable statement reached *shrug*") }
        };

        let view_info = vk::ImageViewCreateInfo {
            image: def_image.gpu_image.image,
            format: info.format,
            view_type,
            components: COMPONENT_MAPPING_DEFAULT,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: info.mip_levels,
                base_array_layer: 0,
                layer_count: 1
            },
            ..Default::default()
        };
        def_image.gpu_image.view = Some(self.device.create_image_view(&view_info, MEMORY_ALLOCATOR).unwrap());
        def_image
    }

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
                        println!("\"{}\" was chosen as primary GPU.", name);
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

            let mut extension_names = vec![
                ash::extensions::khr::Swapchain::name().as_ptr(),
                ash::extensions::khr::Synchronization2::name().as_ptr()
            ];
            for extension in vk_instance.enumerate_device_extension_properties(vk_physical_device).expect("Error enumerating device extensions") {
                let ext_name = CStr::from_ptr(extension.extension_name.as_ptr());
                if let Ok(name) = ext_name.to_str() {
                    if name == "VK_KHR_portability_subset" {
                        extension_names.push(CStr::from_bytes_with_nul_unchecked(b"VK_KHR_portability_subset\0").as_ptr());
                    } else if name == "VK_KHR_shader_non_semantic_info" {
                        extension_names.push(CStr::from_bytes_with_nul_unchecked(b"VK_KHR_shader_non_semantic_info\0").as_ptr());
                    }
                }
            }

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
        let vk_ext_sync2 = ash::extensions::khr::Synchronization2::new(&vk_instance, &vk_device);

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
            ext_sync2: vk_ext_sync2,
            main_queue_family_index: queue_family_index,
            command_pool,
            command_buffer_indices: FreeList::with_capacity(command_buffer_count),
            command_buffers: general_command_buffers,
            command_buffer_fence: graphics_command_buffer_fence,
            samplers: DenseSlotMap::with_key()
        }
    }
}

pub struct GPUBufferBlock {
    pub start_offset: u64,       //In f32s,
    pub length: u64              //In f32s
}

pub struct GPUBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    length: vk::DeviceSize
}

impl GPUBuffer {
    //Read-only access to fields
    pub fn buffer(&self) -> vk::Buffer { self.buffer }
    pub fn length(&self) -> vk::DeviceSize { self.length }

    pub fn allocate(gpu: &mut VulkanGraphicsDevice, size: vk::DeviceSize, alignment: vk::DeviceSize, usage_flags: vk::BufferUsageFlags, memory_location: MemoryLocation) -> Self {
        let vk_buffer;
        let actual_size = size_to_alignment!(size, alignment);
        let allocation = unsafe {
            let buffer_create_info = vk::BufferCreateInfo {
                usage: usage_flags,
                size: actual_size,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            vk_buffer = gpu.device.create_buffer(&buffer_create_info, MEMORY_ALLOCATOR).unwrap();
            let mem_reqs = gpu.device.get_buffer_memory_requirements(vk_buffer);

            let a = gpu.allocator.allocate(&AllocationCreateDesc {
                name: "",
                requirements: mem_reqs,
                location: memory_location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged
            }).unwrap();
            gpu.device.bind_buffer_memory(vk_buffer, a.memory(), a.offset()).unwrap();
            a
        };

        GPUBuffer {
            buffer: vk_buffer,
            allocation,
            length: actual_size
        }
    }

    pub fn free(self, gpu: &mut VulkanGraphicsDevice) {
        gpu.allocator.free(self.allocation).unwrap();
        unsafe { gpu.device.destroy_buffer(self.buffer, MEMORY_ALLOCATOR); }
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

    pub fn write_buffer<T>(&self, gpu: &mut VulkanGraphicsDevice, in_buffer: &[T]) {
        self.write_subbuffer_elements(gpu, in_buffer, 0);
    }

    pub fn write_subbuffer_elements<T>(&self, gpu: &mut VulkanGraphicsDevice, in_buffer: &[T], offset: u64) {
        let byte_buffer = slice_to_bytes(in_buffer);
        let byte_offset = offset * size_of::<T>() as u64;
        self.write_subbuffer_bytes(gpu, byte_buffer, byte_offset as u64);
    }

    pub fn write_subbuffer_bytes(&self, gpu: &mut VulkanGraphicsDevice, in_buffer: &[u8], offset: u64) {
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
                    upload_GPU_buffer(gpu, self.buffer, offset, in_buffer);
                }
            }
        }
    }
}