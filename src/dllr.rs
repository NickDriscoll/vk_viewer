use std::{ffi::c_void, io::Read};

use ash::vk;
use sdl2::video::Window;
use std::ptr;
use crate::*;

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

pub unsafe fn allocate_buffer_memory(vk_instance: &ash::Instance, vk_physical_device: vk::PhysicalDevice, vk_device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceMemory {
    let mem_reqs = vk_device.get_buffer_memory_requirements(buffer);
    let memory_type_index = get_memory_type_index(
        &vk_instance,
        vk_physical_device,
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
    vk_device.allocate_memory(&alloc_info, VK_MEMORY_ALLOCATOR).unwrap()    
}

pub unsafe fn allocate_image_memory(vk_instance: &ash::Instance, vk_physical_device: vk::PhysicalDevice, vk_device: &ash::Device, image: vk::Image) -> vk::DeviceMemory {
    let mem_reqs = vk_device.get_image_memory_requirements(image);

    //Search for the largest DEVICE_LOCAL heap the device advertises
    let memory_type_index = get_memory_type_index(&vk_instance, vk_physical_device, mem_reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL);
    if let None = memory_type_index {
        crash_with_error_dialog("Image memory allocation failed.");
    }
    let memory_type_index = memory_type_index.unwrap();

    let allocate_info = vk::MemoryAllocateInfo {
        allocation_size: mem_reqs.size,
        memory_type_index,
        ..Default::default()
    };
    vk_device.allocate_memory(&allocate_info, VK_MEMORY_ALLOCATOR).unwrap()
}

pub unsafe fn load_shader_stage(vk_device: &ash::Device, shader_stage_flags: vk::ShaderStageFlags, path: &str) -> vk::PipelineShaderStageCreateInfo {
    let mut file = File::open(path).unwrap();
    let spv = ash::util::read_spv(&mut file).unwrap();

    let module_create_info = vk::ShaderModuleCreateInfo {
        code_size: spv.len() * size_of::<u32>(),
        p_code: spv.as_ptr(),
        ..Default::default()
    };
    let module = vk_device.create_shader_module(&module_create_info, VK_MEMORY_ALLOCATOR).unwrap();

    vk::PipelineShaderStageCreateInfo {
        stage: shader_stage_flags,
        p_name: "main\0".as_ptr() as *const i8,
        module,
        ..Default::default()
    }
}

pub unsafe fn load_bc7_texture(
    vk: &VulkanAPI,
    vk_command_buffer: vk::CommandBuffer,
    width: u32,
    height: u32,
    path: &str
) -> vk::Image {        
    const BC7_HEADER_SIZE: usize = 148;
    let file = unwrap_result(File::open(path));
    let raw_bytes: Vec<u8> = file.bytes().map(|n|{ n.unwrap() }).collect();
    let raw_bytes = &raw_bytes[BC7_HEADER_SIZE..];

    let size = (raw_bytes.len()) as vk::DeviceSize;

    let buffer_create_info = vk::BufferCreateInfo {
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        size,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let staging_buffer = vk.device.create_buffer(&buffer_create_info, VK_MEMORY_ALLOCATOR).unwrap();            
    let staging_buffer_memory = dllr::allocate_buffer_memory(&vk.instance, vk.physical_device, &vk.device, staging_buffer);    
    vk.device.bind_buffer_memory(staging_buffer, staging_buffer_memory, 0).unwrap();

    let staging_ptr = vk.device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty()).unwrap();
    ptr::copy_nonoverlapping(raw_bytes.as_ptr(), staging_ptr as *mut _, size as usize);
    vk.device.unmap_memory(staging_buffer_memory);
    
    let image_extent = vk::Extent3D {
        width,
        height,
        depth: 1
    };
    let image_create_info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::BC7_SRGB_BLOCK,
        extent: image_extent,
        mip_levels: 10,
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
    let image = vk.device.create_image(&image_create_info, VK_MEMORY_ALLOCATOR).unwrap();

    let image_memory = dllr::allocate_image_memory(&vk.instance, vk.physical_device, &vk.device, image);

    vk.device.bind_image_memory(image, image_memory, 0).unwrap();

    vk.device.begin_command_buffer(vk_command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 10,
        base_array_layer: 0,
        layer_count: 1
    };
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        image,
        subresource_range,
        ..Default::default()
    };
    vk.device.cmd_pipeline_barrier(vk_command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_memory_barrier]);

    let subresource_layers = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1

    };
    let copy_region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_extent,
        image_offset: vk::Offset3D::default(),
        image_subresource: subresource_layers
    };
    vk.device.cmd_copy_buffer_to_image(vk_command_buffer, staging_buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy_region]);
    
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1
    };
    let image_memory_barrier = vk::ImageMemoryBarrier {
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        image,
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

    let fence = vk.device.create_fence(&vk::FenceCreateInfo::default(), VK_MEMORY_ALLOCATOR).unwrap();
    let queue = vk.device.get_device_queue(vk.queue_family_index, 0);
    vk.device.queue_submit(queue, &[submit_info], fence).unwrap();
    vk.device.wait_for_fences(&[fence], true, vk::DeviceSize::MAX).unwrap();
    vk.device.destroy_fence(fence, VK_MEMORY_ALLOCATOR);
    vk.device.destroy_buffer(staging_buffer, VK_MEMORY_ALLOCATOR);

    image
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
        
        //Initialize the Vulkan API
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

            unsafe { vk_entry.create_instance(&vk_create_info, VK_MEMORY_ALLOCATOR).unwrap() }
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

                    vk_instance.create_device(vk_physical_device, &create_info, VK_MEMORY_ALLOCATOR).unwrap()
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
    pub fn backing_buffer(&self) -> vk::Buffer { self.backing_buffer }
    pub fn current_offset(&self) -> u64 { self.current_offset }
    pub fn max_size(&self) -> u64 { self.max_size }

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

pub struct FrameUniforms {
    view_from_world: glm::TMat4<f32>,
    clip_from_view: glm::TMat4<f32>,
    clip_from_screen: glm::TMat4<f32>
}